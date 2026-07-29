"""Independent evidence-only verifier for T-RISK-20260728-04.

This module intentionally depends only on the Python standard library and
NumPy.  In particular it does not import the producer, physics backends, the
production statistics implementation, or any earlier verifier.  It treats
the immutable receipts and content-addressed objects as hostile input.

The verifier has three deliberately separate layers:

* :func:`audit_transaction` proves the closed 518-cell raw denominator,
  lineage, hashes, seeds, state-major fault ordering, RESET sidecars and
  continuation links;
* :func:`materialize_gate` maps every row of the frozen 3,043-gate blueprint
  to observed cluster contributions (including six-state PTM blocks and the
  two independent cutoff legs);
* :func:`joint_maxt` independently implements the shared-root B=199
  Rademacher maxT calculation.  Density contributions are evaluated in
  blocks so a 4,608 x 132 x 132 tensor is never materialised.

No external scientific claim is emitted by this module.  Even a successful
internal twin-equivalence verdict leaves all external/Puviani/SOTA/LER/
lifetime/hardware/twin claim fields ``null``.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


TASK_ID = "T-RISK-20260728-04"
SCHEMA = "PHASE9-POWERED-TWIN-INDEPENDENT-VERIFICATION-V1"
CLAIM_FIELDS = (
    "twin_qualification",
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "rank",
    "hardware_measured",
)
HEX = frozenset("0123456789abcdef")
LABELS = ("0", "1", "+", "-", "+i", "-i")
# Frozen independently from the powered producer.  CSV column membership and
# order are part of the evidence schema; permissive subset parsing would allow
# an unreviewed producer schema to enter a formal transaction.
LEDGER_COLUMNS = (
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
    "logical_block_11_real", "logical_block_11_imag", "density_trace_error",
    "density_hermiticity_frobenius", "density_minimum_eigenvalue",
    "density_quantization_frobenius_error",
    "density_quantization_certified_frobenius_bound",
    "density_quantization_trace_distance_bound",
    "posterior_normalization_error", "level_normalization_error",
    "reference_posterior_l1_error", "reference_log_evidence_error",
    "conservation_pass", "exception_type", "exception_message",
    "cluster_root_id", "physical_seed_address", "heldout_seed_address",
    "primary_reset_estimand", "sampled_reset_nonvoting",
    "pre_reset_causal_receipt_sha256", "fault_state_index",
    "fault_within_state_index", "pre_intervention_state_sha256",
    "intervention_delta_sha256", "intervention_applied",
    "pre_intervention_drift_0", "pre_intervention_drift_1",
    "pre_intervention_drift_2", "pre_intervention_drift_3",
    "pre_intervention_drift_4", "input_intervention_drift_0",
    "input_intervention_drift_1", "input_intervention_drift_2",
    "input_intervention_drift_3", "input_intervention_drift_4",
    "pre_intervention_non_drift_state_sha256",
    "input_non_drift_state_sha256",
    "intervention_application_receipt_sha256", "input_state_sha256",
    "input_evaluator_sha256", "output_state_sha256",
    "output_evaluator_sha256", "expected_reset_ancestor_receipt_sha256",
)


class EvidenceIncomplete(RuntimeError):
    """Any condition that makes the formal evidence non-voting."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    if not path.is_file() or path.is_symlink():
        raise EvidenceIncomplete(f"missing/non-regular JSON: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceIncomplete(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceIncomplete(f"JSON root is not an object: {path}")
    return value


def _safe_inside(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise EvidenceIncomplete(f"{label} has unsafe path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceIncomplete(f"{label} escapes repository") from exc
    return candidate


def _hex_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in HEX for char in value)
    ):
        raise EvidenceIncomplete(f"{label} is not a lowercase SHA-256")
    return value


def _bool(text: str, label: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise EvidenceIncomplete(f"{label} is not a canonical boolean")


def _float(text: str, label: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise EvidenceIncomplete(f"{label} is not numeric") from exc
    if not math.isfinite(value):
        raise EvidenceIncomplete(f"{label} is non-finite")
    return value


def _int(text: str, label: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise EvidenceIncomplete(f"{label} is not integral") from exc
    if str(value) != str(text):
        raise EvidenceIncomplete(f"{label} is not canonical integer")
    return value


def rademacher_sign(seed: int, replicate: int, root_id: str) -> int:
    if (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or isinstance(replicate, bool) or not isinstance(replicate, int)
        or replicate < 0 or not isinstance(root_id, str) or not root_id
    ):
        raise ValueError("invalid Rademacher address")
    digest = sha256(f"{seed}|{replicate}|{root_id}".encode("utf-8")).digest()
    return 1 if digest[0] & 1 else -1


def rademacher_matrix(seed: int, roots: Sequence[str], replicates: int = 199) -> np.ndarray:
    if replicates != 199 or not roots or len(set(roots)) != len(roots):
        raise ValueError("formal signs require B=199 and unique roots")
    return np.asarray(
        [[rademacher_sign(seed, b, root) for root in roots]
         for b in range(replicates)],
        dtype=np.int8,
    )


def _functional(kind: str, value: np.ndarray | float | complex) -> float:
    array = np.asarray(value)
    if kind == "scalar_abs_mean":
        return abs(float(np.real(array)))
    if kind == "scalar_nonnegative_mean":
        # Observed contributions are validated nonnegative at material
        # construction.  Multiplier pseudo-means are signed perturbations and
        # must not be truncated or rejected.
        return float(np.real(array))
    if kind == "vector_l1_mean":
        return float(np.linalg.norm(array, ord=1))
    if kind == "vector_l2_mean":
        return float(np.linalg.norm(array, ord=2))
    if kind == "vector_linf_mean":
        return float(np.linalg.norm(array, ord=np.inf))
    if kind == "density_half_trace_mean":
        hermitian = (array + array.conj().T) / 2.0
        return float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))) / 2.0)
    raise EvidenceIncomplete(f"unsupported functional {kind}")


@dataclass(frozen=True)
class Material:
    gate_id: str
    family: str
    direction: str
    margin: float
    kind: str
    roots: tuple[str, ...]
    value_shape: tuple[int, ...]
    blocks: Callable[[], Iterator[tuple[int, np.ndarray]]]
    quantization_radius: float = 0.0
    leg: str = ""


@dataclass(frozen=True)
class GateEvaluation:
    gate_id: str
    family: str
    direction: str
    margin: float
    kind: str
    cluster_count: int
    quantization_radius: float
    leg: str
    point: float
    standard_error: float
    standardized: np.ndarray
    replicate_digest: str


def array_material(
    *, gate_id: str, family: str, direction: str, margin: float, kind: str,
    roots: Sequence[str], values: np.ndarray, quantization_radius: float = 0.0,
    leg: str = "",
) -> Material:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[0] != len(roots):
        raise EvidenceIncomplete("array material denominator mismatch")
    if not np.all(np.isfinite(array)):
        raise EvidenceIncomplete("array material contains non-finite values")
    if kind == "scalar_nonnegative_mean" and np.any(array < 0.0):
        raise EvidenceIncomplete("nonnegative observed material contains negatives")
    if (
        not math.isfinite(float(quantization_radius))
        or float(quantization_radius) < 0.0
    ):
        raise EvidenceIncomplete("material quantization radius invalid")
    return Material(
        gate_id, family, direction, float(margin), kind, tuple(roots),
        tuple(array.shape[1:]), lambda: iter(((0, array),)),
        float(quantization_radius), leg,
    )


def evaluate_material(material: Material, signs: np.ndarray) -> GateEvaluation:
    """Two-pass observed multiplier evaluation, including streamed densities."""

    n = len(material.roots)
    if (
        not math.isfinite(material.quantization_radius)
        or material.quantization_radius < 0.0
    ):
        raise EvidenceIncomplete("material quantization radius invalid")
    if n < 2 or signs.shape != (199, n):
        raise EvidenceIncomplete("gate denominator/sign shape mismatch")
    total: np.ndarray | None = None
    seen = 0
    for start, block in material.blocks():
        value = np.asarray(block)
        if start != seen or value.shape[1:] != material.value_shape:
            raise EvidenceIncomplete("material block coverage/shape mismatch")
        if not np.all(np.isfinite(value)):
            raise EvidenceIncomplete("material block is non-finite")
        piece = np.sum(value, axis=0, dtype=np.complex128 if np.iscomplexobj(value) else np.float64)
        total = piece if total is None else total + piece
        seen += value.shape[0]
    if total is None or seen != n:
        raise EvidenceIncomplete("material blocks do not close denominator")
    mean = total / n
    point = _functional(material.kind, mean)
    perturb = np.zeros((199, *material.value_shape), dtype=np.complex128 if np.iscomplexobj(total) else np.float64)
    seen = 0
    for start, block in material.blocks():
        value = np.asarray(block)
        if start != seen:
            raise EvidenceIncomplete("material second-pass ordering drift")
        centered = (value - mean).reshape(value.shape[0], -1)
        update = signs[:, start:start + value.shape[0]].astype(np.float64) @ centered / n
        perturb += update.reshape((199, *material.value_shape))
        seen += value.shape[0]
    replicate = np.asarray(
        [_functional(material.kind, mean + delta) for delta in perturb],
        dtype=np.float64,
    )
    errors = np.abs(replicate - point)
    se = float(np.sqrt(np.mean(np.square(errors))))
    if not math.isfinite(se) or se <= 1e-12:
        raise EvidenceIncomplete(f"degenerate observed SE: {material.gate_id}")
    return GateEvaluation(
        material.gate_id, material.family, material.direction,
        material.margin, material.kind, len(material.roots),
        material.quantization_radius, material.leg, point, se, errors / se,
        sha256(np.asarray(replicate, dtype="<f8").tobytes()).hexdigest(),
    )


def _higher(values: np.ndarray, probability: float = .95) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.shape != (199,) or not np.all(np.isfinite(ordered)):
        raise EvidenceIncomplete("maxT maxima are invalid")
    return float(ordered[max(0, min(198, math.ceil(probability * 199) - 1))])


def joint_maxt(
    gates: Iterable[Material | tuple[Material, Material]],
    *, expected_gate_ids: Sequence[str], seed: int,
    expected_count: int = 3037,
) -> dict[str, Any]:
    expected = tuple(expected_gate_ids)
    if len(expected) != expected_count or len(set(expected)) != expected_count:
        raise EvidenceIncomplete("closed stochastic gate universe drift")
    evaluations: list[tuple[GateEvaluation, GateEvaluation | None]] = []
    observed: set[str] = set()
    sign_cache: dict[tuple[str, ...], np.ndarray] = {}
    for item in gates:
        legs = item if isinstance(item, tuple) else (item,)
        if len(legs) not in {1, 2}:
            raise EvidenceIncomplete("gate has invalid leg count")
        first = legs[0]
        if first.gate_id in observed or any(leg.gate_id != first.gate_id for leg in legs):
            raise EvidenceIncomplete("duplicate/mismatched stochastic gate")
        if any(leg.roots != first.roots for leg in legs):
            raise EvidenceIncomplete("cutoff legs do not share frozen roots")
        observed.add(first.gate_id)
        signs = sign_cache.setdefault(first.roots, rademacher_matrix(seed, first.roots))
        values = tuple(evaluate_material(leg, signs) for leg in legs)
        evaluations.append((values[0], values[1] if len(values) == 2 else None))
    if observed != set(expected) or len(evaluations) != expected_count:
        raise EvidenceIncomplete("closed stochastic gate coverage mismatch")
    evaluations.sort(key=lambda pair: pair[0].gate_id)
    standardized = np.stack([
        one.standardized if two is None else np.maximum(one.standardized, two.standardized)
        for one, two in evaluations
    ])
    maxima = np.max(standardized, axis=0)
    critical = _higher(maxima)
    rows: list[dict[str, Any]] = []
    for first, second in evaluations:
        legs = (first,) if second is None else (first, second)
        bounds = []
        for value in legs:
            radius = value.quantization_radius
            if value.direction == "upper":
                bounds.append(value.point + critical * value.standard_error + radius)
            else:
                bounds.append(value.point - critical * value.standard_error - radius)
        bound = max(bounds) if first.direction == "upper" else min(bounds)
        passed = bound <= first.margin if first.direction == "upper" else bound >= first.margin
        rows.append({
            "gate_id": first.gate_id,
            "family": first.family,
            "direction": first.direction,
            "margin": first.margin,
            "kind": first.kind,
            "cluster_count": first.cluster_count,
            "point": max(value.point for value in legs),
            "joint_critical": critical,
            "simultaneous_bound": bound,
            "passed": bool(passed),
            "legs": [{
                "leg": value.leg or None,
                "point": value.point,
                "standard_error": value.standard_error,
                "quantization_radius": value.quantization_radius,
                "replicate_values_sha256": value.replicate_digest,
            } for value in legs],
        })
    return {
        "stochastic_gate_count": len(rows),
        "multiplier_replicates": 199,
        "confidence": .95,
        "quantile_method": "higher",
        "rademacher_seed": seed,
        "joint_critical": critical,
        "maxima_sha256": sha256(np.asarray(maxima, dtype="<f8").tobytes()).hexdigest(),
        "all_stochastic_gates_passed": all(row["passed"] for row in rows),
        "gates": rows,
        "root_group_count": len(sign_cache),
        "synthetic_influence_used": False,
        "family_split_used": False,
        "pointwise_z_substitution_used": False,
        "aggregate_rescue_used": False,
    }


@dataclass
class CellEvidence:
    cell: dict[str, Any]
    receipt: dict[str, Any]
    roles: dict[str, Path]
    rows: dict[str, np.ndarray]


class EvidenceArchive:
    """Hash-audited archive with a bounded decoded-cell LRU."""

    def __init__(
        self, root: Path, config: Mapping[str, Any], plan: Mapping[str, Any],
        receipt_bindings: Mapping[str, tuple[Path, dict[str, Any]]],
        *, cache_size: int = 12,
        analysis_config: Mapping[str, Any] | None = None,
    ):
        self.root = root
        self.config = config
        self.plan = plan
        self.analysis_config = analysis_config if analysis_config is not None else config
        self.receipts = receipt_bindings
        self.cache_size = cache_size
        self.cache: OrderedDict[str, CellEvidence] = OrderedDict()
        self.by_key = {
            (int(cell["cutoff"]), str(cell["layer"]), str(cell["pair_group_id"]), str(cell["backend"])): cell
            for cell in plan["cells"]
        }

    def get(self, cutoff: int, layer: str, pair_group: str, backend: str) -> CellEvidence:
        try:
            cell = self.by_key[(cutoff, layer, pair_group, backend)]
        except KeyError as exc:
            raise EvidenceIncomplete(f"missing cell {cutoff}/{layer}/{pair_group}/{backend}") from exc
        chunk = str(cell["chunk_id"])
        if chunk in self.cache:
            value = self.cache.pop(chunk)
            self.cache[chunk] = value
            return value
        path, receipt = self.receipts[chunk]
        roles = {str(binding["role"]): _safe_inside(self.root, binding["path"], "object") for binding in receipt["objects"]}
        ledger = roles.get("round_ledger_csv")
        if ledger is None:
            raise EvidenceIncomplete("receipt lacks round ledger")
        fields: dict[str, list[str]] = {}
        with ledger.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
                raise EvidenceIncomplete(
                    "ledger header is not the exact frozen schema/order"
                )
            fields = {name: [] for name in reader.fieldnames}
            for row in reader:
                if None in row or set(row) != set(fields):
                    raise EvidenceIncomplete("ledger row width drift")
                for name in fields:
                    fields[name].append(row[name])
        rows = {key: np.asarray(value) for key, value in fields.items()}
        if len(next(iter(rows.values()), ())) != int(cell["expected_rows"]):
            raise EvidenceIncomplete("ledger row denominator drift")
        value = CellEvidence(dict(cell), receipt, roles, rows)
        self.cache[chunk] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value


def _object_binding(root: Path, binding: Mapping[str, Any]) -> Path:
    required = {
        "schema_version", "role", "media_type", "path", "bytes", "sha256",
        "file_fsync", "directory_fsync", "reopened_and_rehashed",
    }
    if set(binding) != required:
        raise EvidenceIncomplete("object binding key drift")
    digest = _hex_digest(binding.get("sha256"), "object digest")
    path = _safe_inside(root, binding.get("path"), "object")
    if path.name != digest or path.parent.name != digest[:2]:
        raise EvidenceIncomplete("object is not at its content address")
    size, live = _file_sha(path)
    if (
        size != binding.get("bytes") or live != digest
        or binding.get("file_fsync") is not True
        or binding.get("reopened_and_rehashed") is not True
        or not isinstance(binding.get("directory_fsync"), bool)
    ):
        raise EvidenceIncomplete("object binding/live bytes mismatch")
    return path


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if claimed != _sha(unsigned):
        raise EvidenceIncomplete(f"{label} self hash drift")


def _verify_effective_blueprint(
    root: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    effective_binding = config["effective_blueprint_binding"]
    effective_path = _safe_inside(root, effective_binding["path"], "effective blueprint")
    size, digest = _file_sha(effective_path)
    if size != effective_binding["bytes"] or digest != effective_binding["sha256"]:
        raise EvidenceIncomplete("effective blueprint raw binding drift")
    effective = _strict_json(effective_path)
    _verify_self_hash(effective, "analysis_sha256", "effective blueprint")
    if effective.get("analysis_sha256") != effective_binding["analysis_sha256"]:
        raise EvidenceIncomplete("effective blueprint analysis binding drift")
    parent_binding = config["blueprint_binding"]
    parent_path = _safe_inside(root, parent_binding["path"], "selected blueprint")
    parent_size, parent_digest = _file_sha(parent_path)
    if parent_size != parent_binding["bytes"] or parent_digest != parent_binding["sha256"]:
        raise EvidenceIncomplete("selected blueprint raw binding drift")
    parent = _strict_json(parent_path)
    before = parent.get("gates")
    after = effective.get("gates")
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != 3043 or len(after) != 3043:
        raise EvidenceIncomplete("blueprint gate count drift")
    changed: list[tuple[int, str]] = []
    allowed = {"cluster_count", "cluster_scope", "stage"}
    for index, (old, new) in enumerate(zip(before, after)):
        if old.get("gate_id") != new.get("gate_id"):
            raise EvidenceIncomplete("effective blueprint gate ID/reordering drift")
        fields = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
        if fields:
            if fields != allowed or not str(new.get("gate_id", "")).startswith("cutoff/") or "/fault/" not in str(new["gate_id"]):
                raise EvidenceIncomplete("effective blueprint amendment escaped whitelist")
            if (
                old.get("cluster_count") != 1536
                or new.get("cluster_count") != 4608
                or old.get("stage") != "round"
                or new.get("stage") != "trajectory"
                or not str(new.get("cluster_scope", "")).endswith("/all_states")
            ):
                raise EvidenceIncomplete("effective blueprint amendment values drift")
            changed.append((index, str(new["gate_id"])))
    declared = effective.get("changes")
    if (
        len(changed) != 16 or effective.get("changed_gate_count") != 16
        or not isinstance(declared, list) or len(declared) != 16
        or [int(row["gate_index"]) for row in declared] != [index for index, _ in changed]
        or effective.get("scientific_margin_changed") is not False
        or effective.get("gate_deleted") is not False
        or effective.get("gate_id_changed") is not False
        or effective.get("postselection_used") is not False
        or effective.get("cross_state_averaging_used") is not False
        or effective.get("qualified_claim") is not None
    ):
        raise EvidenceIncomplete("effective blueprint 16-change declaration drift")
    return effective


def _verify_release_chain(
    root: Path, config_path: Path, config: Mapping[str, Any],
    plan: Mapping[str, Any], inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    config_size, config_sha = _file_sha(config_path)
    del config_size
    paths = config["artifact_paths"]
    seal = _strict_json(_safe_inside(root, paths["preformal_seal"], "preformal seal"))
    _verify_self_hash(seal, "analysis_sha256", "preformal seal")
    if (
        seal.get("verdict") != "PASS_PREFORMAL_RELEASE"
        or seal.get("raw_execution_released") is not True
        or seal.get("scientific_verdict_released") is not False
        or seal.get("formal_outcomes_accessed") is not False
        or seal.get("config_sha256") != config_sha
        or seal.get("plan_sha256") != plan.get("canonical_plan_sha256")
        or seal.get("scientific_verdict") is not None
        or seal.get("qualified_claim") is not None
        or set(seal.get("claim_boundary", {}).values()) != {None}
    ):
        raise EvidenceIncomplete("preformal seal release/lineage drift")
    snapshot = seal.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise EvidenceIncomplete("seal source snapshot missing")
    _verify_self_hash(snapshot, "source_snapshot_sha256", "source snapshot")
    if (
        snapshot.get("source_snapshot_sha256")
        != seal.get("source_snapshot_sha256")
        or snapshot.get("config", {}).get("sha256") != config_sha
    ):
        raise EvidenceIncomplete("sealed source/config snapshot drift")
    bindings = snapshot.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != (
        int(snapshot.get("runtime_source_count", -1))
        + int(snapshot.get("validation_source_count", -1))
    ):
        raise EvidenceIncomplete("sealed source binding count drift")
    seen_sources: set[str] = set()
    runtime_expected = set(config["runtime_sources"]["paths"])
    validation_expected = set(config["runtime_sources"]["validation_paths"])
    if runtime_expected & validation_expected:
        raise EvidenceIncomplete("config runtime/validation source overlap")
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256", "role"}:
            raise EvidenceIncomplete("sealed source binding schema drift")
        relative = binding["path"]
        if relative in seen_sources:
            raise EvidenceIncomplete("duplicate sealed source binding")
        seen_sources.add(relative)
        expected_role = (
            "runtime" if relative in runtime_expected
            else "validation" if relative in validation_expected
            else None
        )
        if binding["role"] != expected_role:
            raise EvidenceIncomplete("sealed source path/role registry drift")
        source_path = _safe_inside(root, relative, "sealed source")
        size, digest = _file_sha(source_path)
        if size != binding["bytes"] or digest != binding["sha256"]:
            raise EvidenceIncomplete("live source differs from preformal seal")
    if (
        seen_sources != runtime_expected | validation_expected
        or snapshot.get("runtime_source_count") != len(runtime_expected)
        or snapshot.get("validation_source_count") != len(validation_expected)
    ):
        raise EvidenceIncomplete("sealed source set differs from frozen config")
    manifest_path = _safe_inside(root, paths["execution_manifest"], "execution manifest")
    manifest = _strict_json(manifest_path)
    _verify_self_hash(manifest, "manifest_sha256", "execution manifest")
    inventory_path = _safe_inside(root, paths["inventory"], "inventory")
    inventory_size, inventory_file_sha = _file_sha(inventory_path)
    if manifest.get("inventory") != {
        "path": inventory_path.relative_to(root).as_posix(),
        "bytes": inventory_size,
        "sha256": inventory_file_sha,
    }:
        raise EvidenceIncomplete("manifest/inventory raw binding drift")
    if (
        manifest.get("config_sha256") != config_sha
        or inventory.get("config_sha256") != config_sha
        or manifest.get("plan_sha256") != plan.get("canonical_plan_sha256")
        or inventory.get("plan_sha256") != plan.get("canonical_plan_sha256")
        or manifest.get("run_id") != inventory.get("run_id")
        or manifest.get("raw_status") != "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT"
        or manifest.get("scientific_verdict") is not None
        or manifest.get("qualified_claim") is not None
        or set(manifest.get("claim_boundary", {}).values()) != {None}
    ):
        raise EvidenceIncomplete("manifest/inventory lineage or claim drift")
    return seal, manifest, config_sha


def _expected_seed(config: Mapping[str, Any], cell: Mapping[str, Any], position: int, round_index: int) -> tuple[int, int]:
    registry = config["seed_registry"]
    maximum = int(registry["maximum_cluster_positions"])
    physical = (
        int(registry["physical"]["start"])
        + (0 if cell["backend"] == "A" else 1) * 97 * maximum
        + int(cell["pair_group_index"]) * maximum + position
    )
    heldout = (
        int(registry["heldout"]["start"])
        + int(cell["pair_group_index"]) * maximum * int(registry["maximum_horizon"])
        + position * int(registry["maximum_horizon"]) + round_index
    )
    return physical, heldout


def _expected_root(cell: Mapping[str, Any], position: int) -> str:
    if cell["layer"] == "fault":
        state, within = divmod(position, 768)
        return f"{cell['pair_group_id']}/state={LABELS[state]}/cluster={within:04d}"
    return f"{cell['pair_group_id']}/cluster={position:04d}"


def _decode_fixed_ascii(array: np.ndarray) -> list[str]:
    return [bytes(value).rstrip(b"\0").decode("ascii") for value in np.asarray(array)]


def _validate_reset_sidecar(
    evidence: CellEvidence, physics: Mapping[str, Any] | None = None,
) -> None:
    rows = evidence.rows
    reset_indices = np.flatnonzero(rows["reset_requested"] == "True")
    expected = len(reset_indices)
    diagnostics = evidence.receipt["diagnostics"]
    if int(diagnostics["reset_rows"]) != expected or int(diagnostics["reset_sidecar_rows"]) != expected:
        raise EvidenceIncomplete("RESET denominator/sidecar receipt drift")
    rb_roles = {name: path for name, path in evidence.roles.items() if name.startswith("rb_")}
    if expected == 0:
        if rb_roles:
            raise EvidenceIncomplete("non-RESET cell unexpectedly has RB objects")
        return
    required = {
        "rb_valid_npy", "rb_row_index_npy", "rb_success_probability_npy",
        "rb_success_present_npy", "rb_failure_present_npy",
        "rb_conditional_success_density_npy", "rb_conditional_failure_density_npy",
        "rb_sampled_stress_density_npy", "rb_sampled_hidden_outcome_npy",
        "rb_sampled_reset_ack_npy", "rb_branch_trace_distance_npy",
        "rb_sampled_match_trace_distance_npy", "rb_pre_reset_receipt_npy",
        "rb_expected_density_npy",
    }
    if set(rb_roles) != required:
        raise EvidenceIncomplete(f"RESET sidecar role coverage drift: {sorted(set(rb_roles)^required)}")
    arrays = {name: np.load(path, allow_pickle=False, mmap_mode="r") for name, path in rb_roles.items()}
    try:
        dimension = 3 * int(evidence.cell["cutoff"])
        expected_specs = {
            "rb_valid_npy": ((expected,), np.dtype("?")),
            "rb_row_index_npy": ((expected,), np.dtype("<i8")),
            "rb_success_probability_npy": ((expected,), np.dtype("<f8")),
            "rb_success_present_npy": ((expected,), np.dtype("?")),
            "rb_failure_present_npy": ((expected,), np.dtype("?")),
            "rb_conditional_success_density_npy": (
                (expected, dimension, dimension), np.dtype("<c8")
            ),
            "rb_conditional_failure_density_npy": (
                (expected, dimension, dimension), np.dtype("<c8")
            ),
            "rb_sampled_stress_density_npy": (
                (expected, dimension, dimension), np.dtype("<c8")
            ),
            "rb_sampled_hidden_outcome_npy": ((expected,), np.dtype("u1")),
            "rb_sampled_reset_ack_npy": ((expected,), np.dtype("S16")),
            "rb_branch_trace_distance_npy": ((expected,), np.dtype("<f8")),
            "rb_sampled_match_trace_distance_npy": (
                (expected,), np.dtype("<f8")
            ),
            "rb_pre_reset_receipt_npy": ((expected,), np.dtype("S64")),
            "rb_expected_density_npy": (
                (expected, dimension, dimension), np.dtype("<c8")
            ),
        }
        if any(
            arrays[name].shape != shape or arrays[name].dtype != dtype
            for name, (shape, dtype) in expected_specs.items()
        ):
            raise EvidenceIncomplete("RESET sidecar shape/dtype drift")
        if any(value.shape[0] != expected for value in arrays.values()):
            raise EvidenceIncomplete("RESET sidecar first dimension drift")
        if not np.all(arrays["rb_valid_npy"]):
            raise EvidenceIncomplete("RESET sidecar contains invalid slots")
        if not np.array_equal(arrays["rb_row_index_npy"], reset_indices):
            raise EvidenceIncomplete("RESET sidecar row indices drift")
        probability = np.asarray(arrays["rb_success_probability_npy"], dtype=np.float64)
        success_present = np.asarray(arrays["rb_success_present_npy"], dtype=bool)
        failure_present = np.asarray(arrays["rb_failure_present_npy"], dtype=bool)
        if (
            not np.all(np.isfinite(probability))
            or np.any((probability > 0.0) != success_present)
            or np.any((probability < 1.0) != failure_present)
            or np.any((probability < 0.0) | (probability > 1.0))
        ):
            raise EvidenceIncomplete("RESET branch presence/probability mismatch")
        ledger_probability = np.asarray([_float(rows["rao_blackwell_reset_success"][index], "RB success") for index in reset_indices])
        if np.max(np.abs(probability - ledger_probability), initial=0.0) > 5e-12:
            raise EvidenceIncomplete("RESET probability differs from ledger")
        if physics is not None:
            reset_e = float(physics["reset_success_e"])
            reset_f = float(physics["reset_success_f"])
            expected_probability = np.asarray([
                _float(rows["pre_reset_g"][index], "pre-reset g")
                + reset_e * _float(rows["pre_reset_e"][index], "pre-reset e")
                + reset_f * _float(rows["pre_reset_f"][index], "pre-reset f")
                for index in reset_indices
            ])
            if np.max(np.abs(probability - expected_probability), initial=0.0) > 5e-12:
                raise EvidenceIncomplete("RESET success probability not RB-derived")
        hidden = np.asarray(arrays["rb_sampled_hidden_outcome_npy"], dtype=np.uint8)
        acknowledgements = _decode_fixed_ascii(
            arrays["rb_sampled_reset_ack_npy"]
        )
        if (
            np.any((hidden != 0) & (hidden != 1))
            or np.any((hidden == 1) & ~success_present)
            or np.any((hidden == 0) & ~failure_present)
            or any(
                acknowledgement
                != ("success" if hidden[index] == 1 else "failure")
                for index, acknowledgement in enumerate(acknowledgements)
            )
        ):
            raise EvidenceIncomplete("RESET sampled branch presence drift")
        branch_distance = np.asarray(
            arrays["rb_branch_trace_distance_npy"], dtype=np.float64
        )
        match_distance = np.asarray(
            arrays["rb_sampled_match_trace_distance_npy"], dtype=np.float64
        )
        if (
            not np.all(np.isfinite(branch_distance))
            or not np.all(np.isfinite(match_distance))
            or np.any(branch_distance < 0.0)
            or np.any(match_distance < 0.0)
            or np.max(match_distance, initial=0.0) > 5e-6
        ):
            raise EvidenceIncomplete("RESET sampled branch/match witness drift")
        for start in range(0, expected, 16):
            stop = min(expected, start + 16)
            success = np.asarray(
                arrays["rb_conditional_success_density_npy"][start:stop],
                dtype=np.complex128,
            )
            failure = np.asarray(
                arrays["rb_conditional_failure_density_npy"][start:stop],
                dtype=np.complex128,
            )
            expected_density = np.asarray(
                arrays["rb_expected_density_npy"][start:stop],
                dtype=np.complex128,
            )
            sampled = np.asarray(
                arrays["rb_sampled_stress_density_npy"][start:stop],
                dtype=np.complex128,
            )
            if not all(
                np.all(np.isfinite(value))
                for value in (success, failure, expected_density, sampled)
            ):
                raise EvidenceIncomplete("RESET density sidecar non-finite")
            mixture = (
                probability[start:stop, None, None] * success
                + (1.0 - probability[start:stop, None, None]) * failure
            )
            selected = np.where(
                hidden[start:stop, None, None] == 1, success, failure
            )
            both = success_present[start:stop] & failure_present[start:stop]
            for local in np.flatnonzero(both):
                difference = success[local] - failure[local]
                hermitian = (difference + difference.conj().T) / 2.0
                direct = float(
                    np.sum(np.abs(np.linalg.eigvalsh(hermitian))) / 2.0
                )
                if abs(
                    direct - branch_distance[start + int(local)]
                ) > 5e-6:
                    raise EvidenceIncomplete(
                        "RESET branch trace-distance witness drift"
                    )
            if np.max(np.abs(mixture - expected_density), initial=0.0) > 5e-6:
                raise EvidenceIncomplete(
                    "RESET expected density is not the RB mixture"
                )
            if np.max(np.abs(sampled - selected), initial=0.0) > 5e-6:
                raise EvidenceIncomplete(
                    "RESET sampled branch/match witness drift"
                )
        causal = _decode_fixed_ascii(arrays["rb_pre_reset_receipt_npy"])
        ledger_causal = [rows["pre_reset_causal_receipt_sha256"][index] for index in reset_indices]
        if causal != ledger_causal or any(len(value) != 64 for value in causal):
            raise EvidenceIncomplete("RESET causal receipt linkage drift")
        if any(rows["reset_hidden_success"][index] != "" or rows["reset_ack"][index] != "marginalized" for index in reset_indices):
            raise EvidenceIncomplete("hidden RESET branch contaminated primary ledger")
        primary_path = evidence.roles.get("primary_density_npy")
        if primary_path is not None:
            primary = np.load(primary_path, allow_pickle=False, mmap_mode="r")
            try:
                for sidecar_index, row_index in enumerate(reset_indices):
                    density_index = int(evidence.rows["density_index"][row_index])
                    if density_index >= 0 and np.max(
                        np.abs(
                            np.asarray(primary[density_index], dtype=np.complex128)
                            - np.asarray(
                                arrays["rb_expected_density_npy"][sidecar_index],
                                dtype=np.complex128,
                            )
                        )
                    ) > 5e-6:
                        raise EvidenceIncomplete("RESET primary density/sidecar binding drift")
            finally:
                mmap = getattr(primary, "_mmap", None)
                if mmap is not None:
                    mmap.close()
    finally:
        for value in arrays.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _fault_delta(
    scenario: str, specification: Mapping[str, Any], round_index: int,
) -> np.ndarray:
    delta = np.asarray(specification["drift_delta"], dtype=np.float64)
    if delta.shape != (5,) or not np.all(np.isfinite(delta)):
        raise EvidenceIncomplete("frozen fault delta invalid")
    value = np.zeros(5, dtype=np.float64)
    if scenario == "step" and round_index == int(specification["change_round"]):
        value += delta
    elif scenario == "telegraph" and round_index % int(specification["period"]) == 0:
        value += delta if (round_index // int(specification["period"])) % 2 == 0 else -delta
    elif scenario == "burst":
        if round_index == int(specification["start_round"]):
            value += delta
        if round_index == int(specification["start_round"]) + int(specification["duration"]):
            value -= delta
    elif scenario == "compound":
        if round_index == int(specification["change_round"]):
            value += delta
        if round_index == int(specification["burst_start"]):
            value += delta
        if round_index == int(specification["burst_start"]) + int(specification["burst_duration"]):
            value -= delta
    return value


def _validate_density_certificate(
    frobenius_error: float, certified: float, trace_bound: float,
    dimension: int,
) -> None:
    expected_trace = 0.5 * math.sqrt(dimension) * certified
    if (
        not all(math.isfinite(value) for value in (
            frobenius_error, certified, trace_bound
        ))
        or min(frobenius_error, certified, trace_bound) < 0.0
        or frobenius_error > certified + 1e-15
        or abs(trace_bound - expected_trace)
        > max(1e-15, 2e-12 * expected_trace)
    ):
        raise EvidenceIncomplete("density quantization certificate algebra drift")


def _validate_retained_densities(
    density: np.ndarray,
    rows: Mapping[str, np.ndarray],
    ledger_indices: np.ndarray,
    *,
    dimension: int,
    block_size: int = 8,
) -> None:
    """Recompute physicality for every retained complex64 density.

    The CSV diagnostics describe the pre-quantization complex128 matrix.  For
    every retained row we independently diagonalize the Hermitian part of the
    stored matrix and close the original/stored relation with the certified
    Frobenius quantization radius via Weyl's inequality.  No row is sampled.
    """

    if (
        density.shape != (len(ledger_indices), dimension, dimension)
        or block_size <= 0
    ):
        raise EvidenceIncomplete("retained density coverage/shape drift")
    original_trace = _numeric_rows(
        rows, "density_trace_error", ledger_indices
    )
    original_hermiticity = _numeric_rows(
        rows, "density_hermiticity_frobenius", ledger_indices
    )
    original_minimum = _numeric_rows(
        rows, "density_minimum_eigenvalue", ledger_indices
    )
    certified = _numeric_rows(
        rows, "density_quantization_certified_frobenius_bound",
        ledger_indices,
    )
    tolerance = 5e-12
    if (
        np.any(original_trace > 5e-8)
        or np.any(original_hermiticity > 5e-8)
        or np.any(original_minimum < -5e-8)
        or np.any(certified < 0.0)
    ):
        raise EvidenceIncomplete("original density physicality witness invalid")
    for start in range(0, len(ledger_indices), block_size):
        stop = min(len(ledger_indices), start + block_size)
        stack = np.asarray(density[start:stop], dtype=np.complex128)
        if not np.all(np.isfinite(stack)):
            raise EvidenceIncomplete("retained density non-finite")
        adjoint = np.swapaxes(stack.conj(), 1, 2)
        hermitian = 0.5 * (stack + adjoint)
        traces = np.trace(stack, axis1=1, axis2=2)
        trace_error = np.abs(traces.real - 1.0) + np.abs(traces.imag)
        hermiticity = np.linalg.norm(
            (stack - adjoint).reshape(stop - start, -1), axis=1
        )
        minimum = np.linalg.eigvalsh(hermitian)[:, 0]
        q = certified[start:stop]
        if (
            np.any(
                trace_error
                > original_trace[start:stop]
                + math.sqrt(dimension) * q + tolerance
            )
            or np.any(
                hermiticity
                > original_hermiticity[start:stop] + 2.0 * q + tolerance
            )
            or np.any(
                np.abs(minimum - original_minimum[start:stop])
                > q + tolerance
            )
            or np.any(minimum < -5e-8 - q - tolerance)
        ):
            raise EvidenceIncomplete(
                "retained density full physicality/Weyl audit failed"
            )


def _numeric_rows(
    rows: Mapping[str, np.ndarray], field: str, indices: np.ndarray,
) -> np.ndarray:
    try:
        return np.asarray(
            [_float(rows[field][int(index)], field) for index in indices],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise EvidenceIncomplete(f"ledger field missing: {field}") from exc


def _validate_ledger(
    config: Mapping[str, Any], evidence: CellEvidence,
    *, fault_specs: Mapping[str, Any] | None = None,
    physics: Mapping[str, Any] | None = None,
) -> None:
    cell, rows = evidence.cell, evidence.rows
    n = int(cell["expected_rows"])
    required = {
        "row_id", "backend", "cutoff", "seed_position", "round_index",
        "cluster_root_id", "physical_seed_address", "heldout_seed_address",
        "conservation_pass", "exception_type", "exception_message",
        "fault_state_index", "fault_within_state_index", "logical_label",
        "input_state_sha256", "input_evaluator_sha256", "output_state_sha256",
        "output_evaluator_sha256", "expected_reset_ancestor_receipt_sha256",
        "reset_requested", "reset_hidden_success", "reset_ack",
        "pre_reset_causal_receipt_sha256", "rao_blackwell_reset_success",
        "action", "terminal_round", "density_index", "archive_row_index",
        "raw_iq_index", "heldout_iq_index", "pre_intervention_state_sha256",
        "heldout_window_sha256",
        "primary_reset_estimand", "sampled_reset_nonvoting",
        "pre_reset_g", "pre_reset_e", "pre_reset_f",
        "density_quantization_frobenius_error",
        "density_quantization_certified_frobenius_bound",
        "density_quantization_trace_distance_bound",
    }
    if cell["layer"] == "fault" and fault_specs is not None:
        required |= {
            "intervention_applied", "intervention_delta_sha256",
            "intervention_application_receipt_sha256",
            "pre_intervention_non_drift_state_sha256",
            "input_non_drift_state_sha256",
            *(f"pre_intervention_drift_{index}" for index in range(5)),
            *(f"input_intervention_drift_{index}" for index in range(5)),
        }
    if not required.issubset(rows):
        raise EvidenceIncomplete("ledger required columns missing")
    sample_count, horizon = int(cell["sample_count"]), int(cell["horizon"])
    if n != sample_count * horizon:
        raise EvidenceIncomplete("cell frozen denominator mismatch")
    seen_ids: set[str] = set()
    density_indices: list[int] = []
    sequence = (
        tuple(config["formal_matrix"]["fault_action_sequences"][cell["scenario"]])
        if cell["layer"] == "fault" else ()
    )
    last_ancestor_by_position: dict[int, str] = {}
    for index in range(n):
        position, round_index = divmod(index, horizon)
        expected_action = (
            sequence[round_index % len(sequence)]
            if cell["layer"] == "fault" else str(cell.get("action", ""))
        )
        expected_terminal = round_index == horizon - 1
        if (
            rows["backend"][index] != cell["backend"]
            or _int(rows["cutoff"][index], "cutoff") != int(cell["cutoff"])
            or _int(rows["seed_position"][index], "position") != position
            or _int(rows["round_index"][index], "round") != round_index
            or rows["cluster_root_id"][index] != _expected_root(cell, position)
            or rows["action"][index] != expected_action
            or _bool(rows["terminal_round"][index], "terminal_round")
            != expected_terminal
            or _bool(rows["reset_requested"][index], "reset_requested")
            != (expected_action == "RESET")
            or _int(rows["archive_row_index"][index], "archive row") != index
            or _int(rows["raw_iq_index"][index], "raw IQ index") != index
            or _int(rows["heldout_iq_index"][index], "heldout IQ index") != index
        ):
            raise EvidenceIncomplete("ledger identity/order/root drift")
        density_index = _int(rows["density_index"][index], "density index")
        expected_density_index = (
            position if cell["density_retention"] == "terminal_rows" and expected_terminal
            else index if cell["density_retention"] == "all_rows"
            else -1
        )
        if density_index != expected_density_index:
            raise EvidenceIncomplete("density_index coverage/order drift")
        if density_index >= 0:
            density_indices.append(density_index)
            frobenius_error = _float(
                rows["density_quantization_frobenius_error"][index],
                "density quantization frobenius error",
            )
            certified = _float(
                rows["density_quantization_certified_frobenius_bound"][index],
                "density certified frobenius bound",
            )
            trace_bound = _float(
                rows["density_quantization_trace_distance_bound"][index],
                "density trace-distance bound",
            )
            _validate_density_certificate(
                frobenius_error, certified, trace_bound,
                3 * int(cell["cutoff"]),
            )
        physical, heldout = _expected_seed(config, cell, position, round_index)
        if (
            _int(rows["physical_seed_address"][index], "physical seed") != physical
            or _int(rows["heldout_seed_address"][index], "heldout seed") != heldout
        ):
            raise EvidenceIncomplete("formal seed address drift")
        row_id = rows["row_id"][index]
        if not row_id or row_id in seen_ids:
            raise EvidenceIncomplete("row IDs are empty/duplicated")
        seen_ids.add(row_id)
        if (
            rows["conservation_pass"][index] != "True"
            or rows["exception_type"][index] or rows["exception_message"][index]
        ):
            raise EvidenceIncomplete("exception/conservation failure in voting row")
        if expected_action == "RESET":
            if (
                rows["primary_reset_estimand"][index]
                != "RAO_BLACKWELLIZED_EXPECTED_POST_RESET_DENSITY_AND_LEVELS_V1"
                or rows["sampled_reset_nonvoting"][index] != "True"
            ):
                raise EvidenceIncomplete("RESET primary/nonvoting contract drift")
        elif (
            rows["primary_reset_estimand"][index] != ""
            or rows["sampled_reset_nonvoting"][index] != "False"
            or rows["pre_reset_causal_receipt_sha256"][index] != ""
            or rows["reset_hidden_success"][index] != "False"
            or rows["reset_ack"][index] != "none"
        ):
            raise EvidenceIncomplete("non-RESET row carries hidden RESET evidence")
        if cell["layer"] != "fault":
            if rows["expected_reset_ancestor_receipt_sha256"][index] != "":
                raise EvidenceIncomplete("non-fault RESET ancestor must remain empty")
        if cell["layer"] == "fault":
            for field in (
                "pre_intervention_state_sha256", "input_state_sha256",
                "input_evaluator_sha256", "output_state_sha256",
                "output_evaluator_sha256",
            ):
                _hex_digest(rows[field][index], f"fault {field}")
            state, within = divmod(position, 768)
            if (
                _int(rows["fault_state_index"][index], "fault state") != state
                or _int(rows["fault_within_state_index"][index], "fault within") != within
                or rows["logical_label"][index] != LABELS[state]
            ):
                raise EvidenceIncomplete("fault state-major 6x768 ordering drift")
            if round_index > 0:
                previous = index - 1
                if (
                    rows["pre_intervention_state_sha256"][index]
                    != rows["output_state_sha256"][previous]
                    or rows["input_evaluator_sha256"][index] != rows["output_evaluator_sha256"][previous]
                ):
                    raise EvidenceIncomplete("fault continuation state/evaluator link drift")
            previous_ancestor = last_ancestor_by_position.get(position, "")
            current_ancestor = rows["expected_reset_ancestor_receipt_sha256"][index]
            if expected_action == "RESET":
                causal = rows["pre_reset_causal_receipt_sha256"][index]
                if not causal or current_ancestor != causal:
                    raise EvidenceIncomplete("RESET row ancestor linkage drift")
                previous_ancestor = causal
            elif current_ancestor != previous_ancestor:
                raise EvidenceIncomplete("persistent RESET ancestor linkage drift")
            last_ancestor_by_position[position] = previous_ancestor
            if fault_specs is not None:
                specification = fault_specs[cell["scenario"]]
                delta = _fault_delta(cell["scenario"], specification, round_index)
                applied = bool(np.any(delta))
                if (
                    _bool(rows["intervention_applied"][index], "intervention_applied")
                    != applied
                    or rows["intervention_delta_sha256"][index]
                    != sha256(
                        np.asarray(delta, dtype="<f8").tobytes(order="C")
                    ).hexdigest()
                ):
                    raise EvidenceIncomplete("fault intervention witness drift")
                pre_drift = np.asarray([
                    _float(
                        rows[f"pre_intervention_drift_{component}"][index],
                        "pre-intervention drift",
                    )
                    for component in range(5)
                ], dtype="<f8")
                input_drift = np.asarray([
                    _float(
                        rows[f"input_intervention_drift_{component}"][index],
                        "input intervention drift",
                    )
                    for component in range(5)
                ], dtype="<f8")
                if not np.array_equal(input_drift, pre_drift + delta):
                    raise EvidenceIncomplete(
                        "fault intervention drift application mismatch"
                    )
                pre_non_drift = _hex_digest(
                    rows["pre_intervention_non_drift_state_sha256"][index],
                    "pre-intervention non-drift state",
                )
                input_non_drift = _hex_digest(
                    rows["input_non_drift_state_sha256"][index],
                    "input non-drift state",
                )
                if input_non_drift != pre_non_drift:
                    raise EvidenceIncomplete(
                        "fault intervention mutated non-drift state"
                    )
                receipt_payload = {
                    "schema": "PHASE9-INTERVENTION-APPLICATION-RECEIPT-V1",
                    "row_id": rows["row_id"][index],
                    "scenario": str(cell["scenario"]),
                    "round_index": round_index,
                    "intervention_delta_sha256": rows[
                        "intervention_delta_sha256"
                    ][index],
                    "intervention_applied": applied,
                    "pre_intervention_drift_hex": [
                        float(value).hex() for value in pre_drift
                    ],
                    "input_intervention_drift_hex": [
                        float(value).hex() for value in input_drift
                    ],
                    "pre_intervention_non_drift_state_sha256": pre_non_drift,
                    "input_non_drift_state_sha256": input_non_drift,
                }
                application = _sha(receipt_payload)
                if (
                    rows["intervention_application_receipt_sha256"][index]
                    != application
                ):
                    raise EvidenceIncomplete(
                        "fault intervention application receipt drift"
                    )
                if not applied and (
                    rows["input_state_sha256"][index]
                    != rows["pre_intervention_state_sha256"][index]
                ):
                    raise EvidenceIncomplete("zero-delta input/pre-intervention drift")
    if density_indices != list(range(len(density_indices))):
        raise EvidenceIncomplete("density indices are duplicated/gapped")
    base_roles = {"round_ledger_csv", "raw_iq_npy", "heldout_iq_npy"}
    if cell["density_retention"] != "none":
        base_roles.add("primary_density_npy")
    reset_count = int(evidence.receipt["diagnostics"]["reset_rows"])
    if reset_count:
        base_roles |= {
            "rb_valid_npy", "rb_row_index_npy", "rb_success_probability_npy",
            "rb_success_present_npy", "rb_failure_present_npy",
            "rb_conditional_success_density_npy",
            "rb_conditional_failure_density_npy",
            "rb_sampled_stress_density_npy",
            "rb_sampled_hidden_outcome_npy", "rb_sampled_reset_ack_npy",
            "rb_branch_trace_distance_npy",
            "rb_sampled_match_trace_distance_npy",
            "rb_pre_reset_receipt_npy", "rb_expected_density_npy",
        }
    anchor = config["formal_matrix"]["mapping_anchor_plan_indices"].get(
        str(cell["cutoff"])
    )
    if anchor == cell.get("plan_index"):
        mapping_roles = {
            "mapping_isometry_a_npy", "mapping_isometry_b_npy",
            "mapping_projector_a_npy", "mapping_projector_b_npy",
        }
        base_roles |= mapping_roles
        for role in mapping_roles:
            try:
                value = np.load(evidence.roles[role], allow_pickle=False)
            except KeyError as exc:
                raise EvidenceIncomplete("mapping anchor object missing") from exc
            if value.ndim != 2 or not np.all(np.isfinite(value)):
                raise EvidenceIncomplete("mapping anchor shape/finiteness drift")
    if not base_roles.issubset(evidence.roles):
        raise EvidenceIncomplete("cell required object roles missing")
    if set(evidence.roles) != base_roles:
        raise EvidenceIncomplete("cell has unknown/unexpected object role")
    for role in ("raw_iq_npy", "heldout_iq_npy"):
        array = np.load(evidence.roles[role], allow_pickle=False, mmap_mode="r")
        try:
            expected_iq_samples = (
                int(physics["iq_samples"]) if physics is not None
                else int(array.shape[1])
            )
            if (
                array.shape != (n, expected_iq_samples, 2)
                or array.dtype != np.dtype("<f8") or not np.all(np.isfinite(array))
            ):
                raise EvidenceIncomplete(f"{role} shape/dtype/finiteness drift")
            if role == "heldout_iq_npy":
                for index in range(n):
                    digest = sha256(
                        np.asarray(array[index], dtype="<f8").tobytes(order="C")
                    ).hexdigest()
                    if rows["heldout_window_sha256"][index] != digest:
                        raise EvidenceIncomplete(
                            "heldout IQ per-row hash/order drift"
                        )
        finally:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
    if cell["density_retention"] != "none":
        density = np.load(
            evidence.roles["primary_density_npy"], allow_pickle=False, mmap_mode="r"
        )
        if cell["density_retention"] == "all_rows":
            ledger_density_indices = np.arange(n, dtype=np.int64)
        elif cell["density_retention"] == "terminal_rows":
            ledger_density_indices = np.arange(
                horizon - 1, n, horizon, dtype=np.int64
            )
        else:
            raise EvidenceIncomplete("unknown density retention policy")
        expected_count = len(ledger_density_indices)
        dimension = 3 * int(cell["cutoff"])
        try:
            if (
                density.shape != (expected_count, dimension, dimension)
                or density.dtype != np.dtype("<c8")
            ):
                raise EvidenceIncomplete("primary density shape/dtype/finiteness drift")
            _validate_retained_densities(
                density, rows, ledger_density_indices,
                dimension=dimension, block_size=8,
            )
        finally:
            mmap = getattr(density, "_mmap", None)
            if mmap is not None:
                mmap.close()
    _validate_reset_sidecar(evidence, physics)


def audit_transaction(
    root: Path, config: Mapping[str, Any], plan: Mapping[str, Any],
    inventory: Mapping[str, Any], *, allow_test_fixture: bool = False,
    config_sha256: str | None = None,
    sealed_source_snapshot_sha256: str | None = None,
    fault_specs: Mapping[str, Any] | None = None,
    physics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], dict[str, Any]]:
    cells = plan.get("cells")
    expected_cells = 2 if allow_test_fixture else 518
    expected_rows = None if allow_test_fixture else 2_085_888
    expected_densities = None if allow_test_fixture else 482_304
    if not isinstance(cells, list) or len(cells) != expected_cells:
        raise EvidenceIncomplete("plan cell count drift")
    if plan.get("canonical_plan_sha256") != _sha(cells):
        raise EvidenceIncomplete("canonical plan hash drift")
    frozen_plan = config.get("plan_contract", {})
    if (
        plan.get("canonical_plan_sha256")
        != frozen_plan.get("canonical_plan_sha256")
        or plan.get("cell_count") != frozen_plan.get("cell_count")
        or plan.get("row_count") != frozen_plan.get("row_count")
        or plan.get("primary_density_count")
        != frozen_plan.get("primary_density_count")
    ):
        raise EvidenceIncomplete("live plan differs from frozen config binding")
    if expected_rows is not None and (
        plan.get("row_count") != expected_rows
        or plan.get("primary_density_count") != expected_densities
    ):
        raise EvidenceIncomplete("plan row/density totals drift")
    if not allow_test_fixture:
        layer_counts = {
            layer: sum(cell.get("layer") == layer for cell in cells)
            for layer in ("shared", "logical", "probe", "fault")
        }
        cutoff_counts = {
            cutoff: sum(cell.get("cutoff") == cutoff for cell in cells)
            for cutoff in (36, 40, 44)
        }
        backend_counts = {
            backend: sum(cell.get("backend") == backend for cell in cells)
            for backend in ("A", "B")
        }
        if (
            layer_counts != {
                "shared": 210, "logical": 252, "probe": 32, "fault": 24
            }
            or cutoff_counts != {36: 194, 40: 162, 44: 162}
            or backend_counts != {"A": 259, "B": 259}
            or [cell.get("plan_index") for cell in cells] != list(range(518))
            or len({cell.get("chunk_id") for cell in cells}) != 518
            or len({cell.get("pair_group_id") for cell in cells}) != 97
            or sum(int(cell.get("expected_rows", -1)) for cell in cells)
            != 2_085_888
            or sum(
                int(cell["sample_count"])
                for cell in cells if cell.get("density_retention") != "none"
            ) != 482_304
        ):
            raise EvidenceIncomplete("frozen plan semantic matrix drift")
        for scenario in ("step", "telegraph", "burst", "compound"):
            selected = [
                cell for cell in cells
                if cell.get("layer") == "fault"
                and cell.get("scenario") == scenario
            ]
            if (
                len(selected) != 6
                or any(
                    cell.get("sample_count") != 4608
                    or cell.get("horizon") != 12
                    or cell.get("expected_rows") != 55_296
                    for cell in selected
                )
            ):
                raise EvidenceIncomplete("frozen fault plan semantic drift")
    binding = config["effective_blueprint_binding"]
    blueprint_path = _safe_inside(root, binding["path"], "effective blueprint")
    size, digest = _file_sha(blueprint_path)
    if size != binding["bytes"] or digest != binding["sha256"]:
        raise EvidenceIncomplete("effective blueprint binding drift")
    if (
        inventory.get("receipt_count") != expected_cells
        or inventory.get("plan_sha256") != plan["canonical_plan_sha256"]
        or inventory.get("raw_status") != "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT"
        or inventory.get("scientific_verdict") is not None
        or inventory.get("qualified_claim") is not None
    ):
        raise EvidenceIncomplete("inventory status/lineage drift")
    unsigned_inventory = dict(inventory)
    claimed = unsigned_inventory.pop("inventory_sha256", None)
    if claimed != _sha(unsigned_inventory):
        raise EvidenceIncomplete("inventory self hash drift")
    listed = inventory.get("receipts")
    if not isinstance(listed, list) or len(listed) != expected_cells:
        raise EvidenceIncomplete("inventory receipt list drift")
    by_chunk = {str(cell["chunk_id"]): cell for cell in cells}
    receipts: dict[str, tuple[Path, dict[str, Any]]] = {}
    receipt_root = _safe_inside(
        root, config["artifact_paths"]["receipt_directory"], "receipt root"
    )
    object_root = _safe_inside(
        root, config["artifact_paths"]["object_store"], "object root"
    )
    totals = {key: 0 for key in (
        "expected_rows", "observed_rows", "exception_rows", "missing_rows",
        "conservation_failures", "reset_rows", "reset_sidecar_rows",
    )}
    source_snapshot: str | None = None
    run_id: str | None = None
    for listed_receipt in listed:
        if not isinstance(listed_receipt, dict) or set(listed_receipt) != {"chunk_id", "receipt_path", "receipt_sha256"}:
            raise EvidenceIncomplete("inventory receipt binding schema drift")
        chunk = listed_receipt["chunk_id"]
        if chunk not in by_chunk or chunk in receipts:
            raise EvidenceIncomplete("unknown/duplicate receipt")
        path = _safe_inside(root, listed_receipt["receipt_path"], "receipt")
        if path != (receipt_root / f"{chunk}.json").resolve():
            raise EvidenceIncomplete("receipt path/chunk content address drift")
        receipt = _strict_json(path)
        claimed_receipt = receipt.get("receipt_sha256")
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        if claimed_receipt != _sha(unsigned) or claimed_receipt != listed_receipt["receipt_sha256"]:
            raise EvidenceIncomplete("receipt self/inventory hash drift")
        if receipt.get("cell") != by_chunk[chunk] or receipt.get("task_id") != TASK_ID:
            raise EvidenceIncomplete("receipt cell/task lineage drift")
        if config_sha256 is not None and receipt.get("config_sha256") != config_sha256:
            raise EvidenceIncomplete("receipt config lineage drift")
        if receipt.get("plan_sha256") != plan["canonical_plan_sha256"]:
            raise EvidenceIncomplete("receipt plan lineage drift")
        source = _hex_digest(receipt.get("source_snapshot_sha256"), "source snapshot")
        source_snapshot = source if source_snapshot is None else source_snapshot
        if source != source_snapshot:
            raise EvidenceIncomplete("mixed source snapshots")
        if (
            sealed_source_snapshot_sha256 is not None
            and source != sealed_source_snapshot_sha256
        ):
            raise EvidenceIncomplete("receipt source snapshot differs from seal")
        run_id = receipt.get("run_id") if run_id is None else run_id
        if receipt.get("run_id") != run_id:
            raise EvidenceIncomplete("mixed run IDs")
        diagnostics = receipt.get("diagnostics")
        if not isinstance(diagnostics, dict) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in diagnostics.values()
        ):
            raise EvidenceIncomplete("receipt diagnostics invalid")
        if (
            diagnostics.get("expected_rows") != int(by_chunk[chunk]["expected_rows"])
            or diagnostics.get("observed_rows") + diagnostics.get("missing_rows") != diagnostics.get("expected_rows")
            or diagnostics.get("exception_rows") != 0
            or diagnostics.get("missing_rows") != 0
            or diagnostics.get("conservation_failures") != 0
            or diagnostics.get("reset_rows") != diagnostics.get("reset_sidecar_rows")
        ):
            raise EvidenceIncomplete("receipt denominator/terminal diagnostic drift")
        roles: set[str] = set()
        for obj in receipt.get("objects", ()):
            if not isinstance(obj, dict) or obj.get("role") in roles:
                raise EvidenceIncomplete("invalid/duplicate object role")
            roles.add(str(obj["role"]))
            object_path = _object_binding(root, obj)
            try:
                object_path.relative_to(object_root)
            except ValueError as exc:
                raise EvidenceIncomplete(
                    "object binding escapes frozen object store"
                ) from exc
        for key in totals:
            totals[key] += int(diagnostics[key])
        receipts[chunk] = (path, receipt)
    if set(receipts) != set(by_chunk):
        raise EvidenceIncomplete("closed receipt coverage mismatch")
    if expected_rows is not None and totals["observed_rows"] != expected_rows:
        raise EvidenceIncomplete("global observed row denominator drift")
    if any(inventory.get("totals", {}).get(key) != value for key, value in totals.items()):
        raise EvidenceIncomplete("inventory/receipt totals mismatch")
    archive = EvidenceArchive(root, config, plan, receipts)
    for cell in cells:
        _validate_ledger(config, archive.get(
            int(cell["cutoff"]), str(cell["layer"]), str(cell["pair_group_id"]), str(cell["backend"])
        ), fault_specs=fault_specs, physics=physics)
    return receipts, {
        "receipt_count": len(receipts),
        "totals": totals,
        "source_snapshot_sha256": source_snapshot,
        "run_id": run_id,
        "fault_state_major_verified": True,
        "reset_sidecars_verified": True,
        "continuation_links_verified": True,
        "retained_density_physicality": {
            "full_coverage_count": plan.get("primary_density_count"),
            "sampled": False,
            "block_size": 8,
            "stored_matrix_eigendecomposition": "numpy.linalg.eigvalsh",
            "original_minimum_eigenvalue_floor": -5e-8,
            "weyl_quantization_certificate_checked": True,
            "trace_and_hermiticity_recomputed": True,
        },
    }


def _numeric(cell: CellEvidence, field: str, indices: np.ndarray) -> np.ndarray:
    try:
        return np.asarray([_float(cell.rows[field][int(i)], field) for i in indices], dtype=np.float64)
    except KeyError as exc:
        raise EvidenceIncomplete(f"ledger field missing: {field}") from exc


def _mapping_projector(
    archive: EvidenceArchive, cutoff: int, backend: str,
) -> np.ndarray:
    anchor = int(
        archive.config["formal_matrix"]["mapping_anchor_plan_indices"][str(cutoff)]
    )
    cell = archive.plan["cells"][anchor]
    evidence = archive.get(
        cutoff, str(cell["layer"]), str(cell["pair_group_id"]),
        str(cell["backend"]),
    )
    role = f"mapping_projector_{backend.lower()}_npy"
    try:
        projector = np.asarray(
            np.load(evidence.roles[role], allow_pickle=False),
            dtype=np.complex128,
        )
    except KeyError as exc:
        raise EvidenceIncomplete(f"mapping projector missing: c{cutoff}/{backend}") from exc
    if projector.shape != (cutoff, cutoff) or not np.all(np.isfinite(projector)):
        raise EvidenceIncomplete("mapping projector shape/finiteness drift")
    return projector


def _density_survival(
    archive: EvidenceArchive, cell: CellEvidence, density_indices: np.ndarray,
) -> np.ndarray:
    density = np.load(
        cell.roles["primary_density_npy"], allow_pickle=False, mmap_mode="r"
    )
    cutoff = int(cell.cell["cutoff"])
    projector = _mapping_projector(archive, cutoff, str(cell.cell["backend"]))
    try:
        values = np.empty(len(density_indices), dtype=np.float64)
        for start in range(0, len(density_indices), 32):
            stop = min(len(density_indices), start + 32)
            stack = np.asarray(
                density[density_indices[start:stop]], dtype=np.complex128
            )
            joint = stack.reshape(stop - start, cutoff, 3, cutoff, 3)
            oscillator = np.trace(joint, axis1=2, axis2=4)
            values[start:stop] = np.einsum(
                "ij,nji->n", projector, oscillator
            ).real
    finally:
        mmap = getattr(density, "_mmap", None)
        if mmap is not None:
            mmap.close()
    if not np.all(np.isfinite(values)):
        raise EvidenceIncomplete("density-derived logical survival non-finite")
    return values


def _tail_values(
    cell: CellEvidence, density_indices: np.ndarray, metric: str,
    quantization_radii: np.ndarray,
) -> np.ndarray:
    density = np.load(
        cell.roles["primary_density_npy"], allow_pickle=False, mmap_mode="r"
    )
    cutoff = int(cell.cell["cutoff"])
    values = np.empty(len(density_indices), dtype=np.float64)
    try:
        for start in range(0, len(density_indices), 32):
            stop = min(len(density_indices), start + 32)
            stack = np.asarray(
                density[density_indices[start:stop]], dtype=np.complex128
            )
            joint = stack.reshape(stop - start, cutoff, 3, cutoff, 3)
            oscillator = np.trace(joint, axis1=2, axis2=4)
            populations = np.diagonal(
                oscillator, axis1=1, axis2=2
            ).real
            if (
                not np.all(np.isfinite(populations))
                or np.min(populations) < -5e-5
                or np.max(np.abs(np.sum(populations, axis=1) - 1.0)) > 5e-5
            ):
                raise EvidenceIncomplete("tail population physicality drift")
            if metric == "normalized_mean_photon":
                values[start:stop] = (
                    populations @ np.arange(cutoff, dtype=np.float64)
                    / float(cutoff - 1)
                )
            elif metric.startswith("top") and metric.endswith("_fock_mass"):
                values[start:stop] = np.sum(
                    populations[:, -int(metric[3]):], axis=1
                )
            elif metric == "commutator_defect":
                values[start:stop] = float(cutoff) * populations[:, -1]
            else:
                raise EvidenceIncomplete(f"unknown tail metric {metric}")
    finally:
        mmap = getattr(density, "_mmap", None)
        if mmap is not None:
            mmap.close()
    multiplier = float(cutoff) if metric == "commutator_defect" else 1.0
    return _project_nonnegative_observable(
        values, np.asarray(quantization_radii, dtype=np.float64) * multiplier
    )


def _project_nonnegative_observable(
    values: np.ndarray, quantization_radii: np.ndarray, *, tolerance: float = 1e-12,
) -> np.ndarray:
    """Certify and project a theoretically nonnegative observable.

    The stored complex64 density may yield a small negative expectation.  A
    row is admissible only when that negativity is covered by its independently
    recorded trace-distance quantization certificate.  Projection onto the
    nonnegative half-line cannot increase error relative to the true,
    nonnegative observable; the original certificate remains in the gate bound.
    """

    observed = np.asarray(values, dtype=np.float64)
    radii = np.asarray(quantization_radii, dtype=np.float64)
    if (
        observed.shape != radii.shape
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(radii))
        or np.any(radii < 0.0)
        or not math.isfinite(float(tolerance))
        or tolerance < 0.0
    ):
        raise EvidenceIncomplete("nonnegative observable certificate invalid")
    if np.any(observed < -radii - tolerance):
        raise EvidenceIncomplete(
            "negative nonnegative-observable value exceeds quantization certificate"
        )
    return np.maximum(observed, 0.0)


def _indices(cell: CellEvidence, *, state: str | None = None, stage: str = "round") -> tuple[np.ndarray, tuple[str, ...]]:
    horizon = int(cell.cell["horizon"])
    sample_count = int(cell.cell["sample_count"])
    positions = np.arange(sample_count, dtype=np.int64)
    if state is not None:
        state_index = LABELS.index(state)
        positions = positions[state_index * 768:(state_index + 1) * 768]
    if horizon == 1:
        indices = positions
    elif stage in {"terminal", "trajectory"}:
        indices = positions * horizon + (horizon - 1)
    else:
        schedule = {
            # Frozen, preregistered contiguous scenario partitions.
            "pre_change": (0, 1, 2, 3),
            "early_post_change": (4, 5, 6, 7),
            "late_post_change": (8, 9, 10, 11),
            "phase_0": (0, 1, 2), "phase_1": (3, 4, 5),
            "phase_2": (6, 7, 8), "phase_3": (9, 10, 11),
            "pre_burst": (0, 1, 2, 3),
            "burst_active": (4, 5, 6),
            # compound also names 6..8 burst_active; selection depends on
            # scenario below rather than one ambiguous global alias.
            "recovery": (7, 8, 9, 10),
            "pre_shift": (0, 1, 2), "shifted": (3, 4, 5),
        }
        if cell.cell["scenario"] == "compound":
            schedule["burst_active"] = (6, 7, 8)
            schedule["recovery"] = (9, 10, 11)
        elif cell.cell["scenario"] == "burst":
            schedule["burst_active"] = (4, 5, 6)
            schedule["recovery"] = (7, 8, 9, 10)
        if stage not in schedule:
            raise EvidenceIncomplete(f"unknown fault stage {stage}")
        # Negative indices encode a stage group and are handled by the caller.
        rounds = np.asarray(schedule[stage], dtype=np.int64)
        indices = (positions[:, None] * horizon + rounds[None, :]).reshape(-1)
    roots = tuple(_expected_root(cell.cell, int(position)) for position in positions)
    return indices, roots


def _stage_reduce(values: np.ndarray, root_count: int) -> np.ndarray:
    if values.shape[0] == root_count:
        return values
    if values.shape[0] % root_count:
        raise EvidenceIncomplete("stage values do not partition trajectories")
    return values.reshape(root_count, values.shape[0] // root_count, *values.shape[1:]).mean(axis=1)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    absolute = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = t * (
        0.319381530 + t * (
            -0.356563782 + t * (
                1.781477937 + t * (-1.821255978 + t * 1.330274429)
            )
        )
    )
    tail = (
        np.exp(-0.5 * absolute * absolute)
        / math.sqrt(2.0 * math.pi) * polynomial
    )
    return np.where(x >= 0.0, 1.0 - tail, tail)


def _validate_probabilities(values: np.ndarray, label: str) -> np.ndarray:
    """Apply the frozen parent probability contract without importing it."""

    probabilities = np.asarray(values, dtype=np.float64)
    if (
        probabilities.shape[-1] != 3
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < -1e-12)
        or np.any(probabilities > 1.0 + 1e-12)
        or np.any(
            np.abs(np.sum(probabilities, axis=-1) - 1.0) > 5e-8
        )
    ):
        raise EvidenceIncomplete(
            f"{label} is not a normalized qutrit probability"
        )
    return np.clip(probabilities, 0.0, 1.0)


def _predictive_inputs(
    archive: EvidenceArchive, cell: CellEvidence, indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    physics = archive.analysis_config["common_physics"]
    base = np.asarray(physics["iq_centers"], dtype=np.float64)
    sigma, count = float(physics["iq_sigma"]), int(physics["iq_samples"])
    if base.shape != (3, 2) or sigma <= 0.0 or count <= 0:
        raise EvidenceIncomplete("frozen IQ physics invalid")
    priors = _validate_probabilities(
        np.stack([
            _numeric(cell, "pre_measurement_g", indices),
            _numeric(cell, "pre_measurement_e", indices),
            _numeric(cell, "pre_measurement_f", indices),
        ], axis=1),
        "pre-measurement prior",
    )
    offsets = np.stack([
        _numeric(cell, "pre_readout_i", indices),
        _numeric(cell, "pre_readout_q", indices),
    ], axis=1)
    return priors, base[None, :, :] + offsets[:, None, :], sigma, count


def _heldout_statistics(
    archive: EvidenceArchive, cell: CellEvidence, indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    priors, centers, sigma, count = _predictive_inputs(archive, cell, indices)
    windows = np.load(
        cell.roles["heldout_iq_npy"], allow_pickle=False, mmap_mode="r"
    )
    try:
        observations = np.asarray(windows[indices], dtype=np.float64)
    finally:
        mmap = getattr(windows, "_mmap", None)
        if mmap is not None:
            mmap.close()
    if observations.shape != (len(indices), count, 2):
        raise EvidenceIncomplete("heldout IQ window shape drift")
    for local, row_index in enumerate(indices):
        digest = sha256(
            np.asarray(observations[local], dtype="<f8").tobytes(order="C")
        ).hexdigest()
        if cell.rows["heldout_window_sha256"][int(row_index)] != digest:
            raise EvidenceIncomplete("heldout window ledger hash drift")
    delta = observations[:, None, :, :] - centers[:, :, None, :]
    logs = (
        -count * math.log(2.0 * math.pi * sigma * sigma)
        - np.sum(delta * delta, axis=(2, 3)) / (2.0 * sigma * sigma)
    )
    weighted = logs + np.log(np.maximum(priors, 1e-300))
    maximum = np.max(weighted, axis=1, keepdims=True)
    posterior = np.exp(weighted - maximum)
    evidence = maximum[:, 0] + np.log(
        np.sum(posterior, axis=1)
    )
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    llr = np.stack((
        logs[:, 0] - logs[:, 1],
        logs[:, 0] - logs[:, 2],
        logs[:, 1] - logs[:, 2],
    ), axis=1) / count
    if not np.all(np.isfinite(posterior)):
        raise EvidenceIncomplete("heldout posterior non-finite")
    return evidence / count, llr, posterior


def _pair_values(
    archive: EvidenceArchive, *, cutoff: int, layer: str, pair: str,
    metric: str, stage: str, state: str | None = None,
) -> tuple[tuple[str, ...], np.ndarray, float]:
    a = archive.get(cutoff, layer, pair, "A")
    b = archive.get(cutoff, layer, pair, "B")
    ia, roots = _indices(a, state=state, stage=stage)
    ib, roots_b = _indices(b, state=state, stage=stage)
    if roots != roots_b or not np.array_equal(ia, ib):
        raise EvidenceIncomplete("paired cell coverage mismatch")
    n = len(roots)
    numerical_radius = 0.0
    scalar_fields = {
        "mean_photon": "mean_photon",
        "mean_photon_difference": "mean_photon",
        "terminal_mean_photon_difference": "mean_photon",
        "logical_survival": "logical_survival",
        "terminal_logical_survival": "logical_survival",
        "rao_blackwell_reset_success": "rao_blackwell_reset_success",
    }
    vector_fields = {
        "level_probability_l1": ("level_g", "level_e", "level_f"),
        "terminal_level_probability_l1": ("level_g", "level_e", "level_f"),
    }
    if metric in {"leakage_residence_rate", "terminal_leakage_residence_rate"}:
        left = np.asarray([
            float(_bool(a.rows["leakage_resident"][int(index)], "leakage"))
            for index in ia
        ])
        right = np.asarray([
            float(_bool(b.rows["leakage_resident"][int(index)], "leakage"))
            for index in ib
        ])
        values = _stage_reduce(left - right, n)
    elif metric == "rao_blackwell_reset_success":
        physics = archive.analysis_config["common_physics"]
        reset_e, reset_f = (
            float(physics["reset_success_e"]),
            float(physics["reset_success_f"]),
        )
        rb_a = (
            _numeric(a, "pre_reset_g", ia)
            + reset_e * _numeric(a, "pre_reset_e", ia)
            + reset_f * _numeric(a, "pre_reset_f", ia)
        )
        rb_b = (
            _numeric(b, "pre_reset_g", ib)
            + reset_e * _numeric(b, "pre_reset_e", ib)
            + reset_f * _numeric(b, "pre_reset_f", ib)
        )
        ledger_a = _numeric(a, "rao_blackwell_reset_success", ia)
        ledger_b = _numeric(b, "rao_blackwell_reset_success", ib)
        if (
            np.max(np.abs(rb_a - ledger_a), initial=0.0) > 5e-12
            or np.max(np.abs(rb_b - ledger_b), initial=0.0) > 5e-12
        ):
            raise EvidenceIncomplete("RESET gate RB/ledger probability drift")
        values = _stage_reduce(rb_a - rb_b, n)
    elif metric in scalar_fields:
        field = scalar_fields[metric]
        scalar_a = _numeric(a, field, ia)
        scalar_b = _numeric(b, field, ib)
        if layer == "fault" and metric == "terminal_logical_survival":
            density_a = _density_survival(
                archive, a, ia // int(a.cell["horizon"])
            )
            density_b = _density_survival(
                archive, b, ib // int(b.cell["horizon"])
            )
            certificate_a = _numeric(
                a, "density_quantization_trace_distance_bound", ia
            )
            certificate_b = _numeric(
                b, "density_quantization_trace_distance_bound", ib
            )
            if (
                np.any(np.abs(scalar_a - density_a) > certificate_a + 1e-6)
                or np.any(np.abs(scalar_b - density_b) > certificate_b + 1e-6)
            ):
                raise EvidenceIncomplete(
                    "fault CSV survival/projector-density cross-check drift"
                )
        values = _stage_reduce(scalar_a - scalar_b, n)
    elif metric in {
        "paired_proper_score_per_sample",
        "terminal_paired_proper_score_per_sample",
        "pairwise_llr_per_sample",
        "posterior_mean_l1",
    }:
        window_a = np.load(a.roles["heldout_iq_npy"], allow_pickle=False, mmap_mode="r")
        window_b = np.load(b.roles["heldout_iq_npy"], allow_pickle=False, mmap_mode="r")
        try:
            if not np.array_equal(window_a[ia], window_b[ib]):
                raise EvidenceIncomplete("A/B common heldout window drift")
        finally:
            for window in (window_a, window_b):
                mmap = getattr(window, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        score_a, llr_a, posterior_a = _heldout_statistics(archive, a, ia)
        score_b, llr_b, posterior_b = _heldout_statistics(archive, b, ib)
        if metric in {
            "paired_proper_score_per_sample",
            "terminal_paired_proper_score_per_sample",
        }:
            values = score_a - score_b
        elif metric == "pairwise_llr_per_sample":
            values = llr_a - llr_b
        else:
            values = posterior_a - posterior_b
        values = _stage_reduce(values, n)
    elif metric in {"predictive_mean_l2", "terminal_predictive_mean_l2"}:
        prior_a, centers_a, _, _ = _predictive_inputs(archive, a, ia)
        prior_b, centers_b, _, _ = _predictive_inputs(archive, b, ib)
        mean_a = np.einsum("nk,nkd->nd", prior_a, centers_a)
        mean_b = np.einsum("nk,nkd->nd", prior_b, centers_b)
        csv_a = np.stack([
            _numeric(a, "predictive_mean_i", ia),
            _numeric(a, "predictive_mean_q", ia),
        ], axis=1)
        csv_b = np.stack([
            _numeric(b, "predictive_mean_i", ib),
            _numeric(b, "predictive_mean_q", ib),
        ], axis=1)
        if (
            np.max(np.abs(mean_a - csv_a), initial=0.0) > 1e-9
            or np.max(np.abs(mean_b - csv_b), initial=0.0) > 1e-9
        ):
            raise EvidenceIncomplete("predictive mean CSV/model drift")
        values = _stage_reduce(mean_a - mean_b, n)
    elif metric in vector_fields:
        fields = vector_fields[metric]
        values = np.stack([_numeric(a, field, ia) - _numeric(b, field, ib) for field in fields], axis=1)
        values = _stage_reduce(values, n)
    elif metric in {"predictive_covariance_frobenius", "terminal_predictive_covariance_frobenius"}:
        def covariance(cell: CellEvidence, indices: np.ndarray) -> np.ndarray:
            prior, centers, sigma, count = _predictive_inputs(
                archive, cell, indices
            )
            mean = np.einsum("nk,nkd->nd", prior, centers)
            centered = centers - mean[:, None, :]
            cov = np.einsum("nk,nki,nkj->nij", prior, centered, centered)
            cov[:, 0, 0] += sigma * sigma / count
            cov[:, 1, 1] += sigma * sigma / count
            csv_cov = np.stack([
                _numeric(cell, "predictive_cov_ii", indices),
                _numeric(cell, "predictive_cov_iq", indices),
                _numeric(cell, "predictive_cov_iq", indices),
                _numeric(cell, "predictive_cov_qq", indices),
            ], axis=1).reshape(len(indices), 2, 2)
            if np.max(np.abs(cov - csv_cov), initial=0.0) > 1e-9:
                raise EvidenceIncomplete("predictive covariance CSV/model drift")
            return cov
        values = covariance(a, ia) - covariance(b, ib)
        values = _stage_reduce(values, n).reshape(n, 4)
    elif metric == "terminal_drift_normalized_l2":
        normalization = np.asarray(
            archive.analysis_config["verification_contract"]["drift_normalization"],
            dtype=np.float64,
        )
        if normalization.shape != (5,) or np.any(normalization <= 0.0):
            raise EvidenceIncomplete("frozen drift normalization invalid")
        values = np.stack([(_numeric(a, f"drift_{k}", ia) - _numeric(b, f"drift_{k}", ib)) / normalization[k] for k in range(5)], axis=1)
        values = _stage_reduce(values, n)
    elif metric in {"predictive_cdf_i_sup", "predictive_cdf_q_sup"}:
        axis = 0 if metric.endswith("i_sup") else 1
        prior_a, centers_a, sigma, count = _predictive_inputs(archive, a, ia)
        prior_b, centers_b, sigma_b, count_b = _predictive_inputs(archive, b, ib)
        if sigma != sigma_b or count != count_b:
            raise EvidenceIncomplete("A/B predictive physics mismatch")
        all_centers = np.concatenate((
            centers_a[:, :, axis].reshape(-1),
            centers_b[:, :, axis].reshape(-1),
        ))
        span = 8.0 * sigma / math.sqrt(count)
        grid = np.linspace(
            float(all_centers.min() - span),
            float(all_centers.max() + span),
            513,
        )
        scale = sigma / math.sqrt(count)
        values = np.empty((len(ia), 513), dtype=np.float64)
        for start in range(0, len(ia), 64):
            end = min(len(ia), start + 64)
            cdf_a = np.sum(
                prior_a[start:end, :, None]
                * _normal_cdf(
                    (grid[None, None, :]
                     - centers_a[start:end, :, axis, None]) / scale
                ),
                axis=1,
            )
            cdf_b = np.sum(
                prior_b[start:end, :, None]
                * _normal_cdf(
                    (grid[None, None, :]
                     - centers_b[start:end, :, axis, None]) / scale
                ),
                axis=1,
            )
            values[start:end] = cdf_a - cdf_b
        values = _stage_reduce(values, n)
        # Independent A&S normal-CDF approximation: <8e-8 per CDF side.
        # A-B supremum receives a conservative 2e-7 external radius.
        numerical_radius = 2e-7
    else:
        raise EvidenceIncomplete(f"unsupported paired metric {metric}")
    # CSV scalar/vector/IQ functionals are computed in float64.  Density
    # certificates are attached only by the density-derived materializers.
    return roots, values, numerical_radius


def _kind(metric: str) -> str:
    if "density" in metric:
        return "density_half_trace_mean"
    if metric in {"level_probability_l1", "terminal_level_probability_l1", "posterior_mean_l1"}:
        return "vector_l1_mean"
    if metric in {"pairwise_llr_per_sample", "predictive_cdf_i_sup", "predictive_cdf_q_sup", "logical_ptm_entry"}:
        return "vector_linf_mean"
    if any(token in metric for token in ("predictive_mean", "predictive_covariance", "drift_normalized")):
        return "vector_l2_mean"
    return "scalar_abs_mean"


def _logical_block_roots(action: str, count: int = 1536) -> tuple[str, ...]:
    if not action or count <= 0:
        raise EvidenceIncomplete("invalid logical composite root request")
    return tuple(
        f"logical/block/{action}/cluster={position:04d}"
        for position in range(count)
    )


def _density_material(
    archive: EvidenceArchive, gate: Mapping[str, Any], *,
    cutoff_a: int, cutoff_b: int, layer: str, pair: str,
    backend_a: str, backend_b: str, state: str | None, stage: str, leg: str = "",
) -> Material:
    left = archive.get(cutoff_a, layer, pair, backend_a)
    right = archive.get(cutoff_b, layer, pair, backend_b)
    il, roots = _indices(left, state=state, stage=stage)
    ir, roots_r = _indices(right, state=state, stage=stage)
    if roots != roots_r or len(il) != len(roots) or len(ir) != len(roots):
        raise EvidenceIncomplete("density gate requires one row per root")
    # Density arrays are indexed by terminal position for faults and row for
    # one-round cells, not necessarily by ledger row index.
    density_l = il // int(left.cell["horizon"]) if int(left.cell["horizon"]) > 1 else il
    density_r = ir // int(right.cell["horizon"]) if int(right.cell["horizon"]) > 1 else ir
    probe_l = np.load(left.roles["primary_density_npy"], allow_pickle=False, mmap_mode="r")
    probe_r = np.load(right.roles["primary_density_npy"], allow_pickle=False, mmap_mode="r")
    try:
        upper_dim = max(probe_l.shape[1], probe_r.shape[1])
    finally:
        for value in (probe_l, probe_r):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def blocks() -> Iterator[tuple[int, np.ndarray]]:
        dl = np.load(left.roles["primary_density_npy"], allow_pickle=False, mmap_mode="r")
        dr = np.load(right.roles["primary_density_npy"], allow_pickle=False, mmap_mode="r")
        try:
            for start in range(0, len(roots), 32):
                end = min(len(roots), start + 32)
                a = np.asarray(dl[density_l[start:end]], dtype=np.complex128)
                b = np.asarray(dr[density_r[start:end]], dtype=np.complex128)
                if a.shape[1] != upper_dim:
                    padded = np.zeros((len(a), upper_dim, upper_dim), dtype=np.complex128)
                    padded[:, :a.shape[1], :a.shape[2]] = a
                    a = padded
                if b.shape[1] != upper_dim:
                    padded = np.zeros((len(b), upper_dim, upper_dim), dtype=np.complex128)
                    padded[:, :b.shape[1], :b.shape[2]] = b
                    b = padded
                yield start, a - b
        finally:
            for value in (dl, dr):
                mmap = getattr(value, "_mmap", None)
                if mmap is not None:
                    mmap.close()

    q = float(np.mean(
        _numeric(left, "density_quantization_trace_distance_bound", il)
        + _numeric(right, "density_quantization_trace_distance_bound", ir)
    ))
    return Material(
        str(gate["gate_id"]), str(gate["family"]), str(gate["direction"]),
        float(gate["margin"]), "density_half_trace_mean", roots,
        (upper_dim, upper_dim), blocks, q, leg,
    )


def _ptm_material(archive: EvidenceArchive, gate: Mapping[str, Any]) -> Material:
    parts = str(gate["gate_id"]).split("/")
    cutoff, action = int(parts[1][1:]), parts[3] if len(parts) == 4 else parts[4]
    roots = _logical_block_roots(action)
    paulis = np.asarray([
        [[1, 0], [0, 1]], [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]], [[1, 0], [0, -1]],
    ], dtype=np.complex128)

    def ptm(backend: str) -> np.ndarray:
        coordinates: dict[str, np.ndarray] = {}
        for label in LABELS:
            cell = archive.get(cutoff, "logical", f"logical/{label}/{action}", backend)
            block = np.stack([
                _numeric(cell, "logical_block_00_real", np.arange(1536)) + 1j * _numeric(cell, "logical_block_00_imag", np.arange(1536)),
                _numeric(cell, "logical_block_01_real", np.arange(1536)) + 1j * _numeric(cell, "logical_block_01_imag", np.arange(1536)),
                _numeric(cell, "logical_block_10_real", np.arange(1536)) + 1j * _numeric(cell, "logical_block_10_imag", np.arange(1536)),
                _numeric(cell, "logical_block_11_real", np.arange(1536)) + 1j * _numeric(cell, "logical_block_11_imag", np.arange(1536)),
            ], axis=1).reshape(1536, 2, 2)
            coordinates[label] = np.einsum("aij,nji->na", paulis, block).real
        matrix = np.zeros((1536, 4, 4), dtype=np.float64)
        identities = []
        for plus, minus, column in (("+", "-", 1), ("+i", "-i", 2), ("0", "1", 3)):
            identities.append((coordinates[plus] + coordinates[minus]) / 2)
            matrix[:, :, column] = (coordinates[plus] - coordinates[minus]) / 2
        matrix[:, :, 0] = np.mean(np.stack(identities), axis=0)
        return matrix.reshape(1536, 16)

    return array_material(
        gate_id=str(gate["gate_id"]), family=str(gate["family"]),
        direction=str(gate["direction"]), margin=float(gate["margin"]),
        kind="vector_linf_mean", roots=roots, values=ptm("A") - ptm("B"),
    )


def evaluate_mapping_arrays(
    metric: str, isometry_a: np.ndarray, isometry_b: np.ndarray,
    projector_a: np.ndarray, projector_b: np.ndarray,
) -> float:
    """Independent exact c36/c40/c44 mapping functional."""

    ia, ib, pa, pb = (
        np.asarray(isometry_a, dtype=np.complex128),
        np.asarray(isometry_b, dtype=np.complex128),
        np.asarray(projector_a, dtype=np.complex128),
        np.asarray(projector_b, dtype=np.complex128),
    )
    if (
        ia.ndim != 2 or ia.shape[1] != 2 or ib.shape != ia.shape or pa.ndim != 2
        or pb.shape != pa.shape or pa.shape[0] != pa.shape[1]
        or pa.shape[0] != ia.shape[0]
        or not all(np.all(np.isfinite(value)) for value in (ia, ib, pa, pb))
    ):
        raise EvidenceIncomplete("mapping array shape/finiteness drift")
    identity = np.eye(2, dtype=np.complex128)
    for name, isometry, projector in (
        ("A", ia, pa), ("B", ib, pb)
    ):
        if (
            np.linalg.norm(isometry.conj().T @ isometry - identity, ord="fro")
            > 1e-8
            or np.linalg.norm(projector - isometry @ isometry.conj().T, ord="fro")
            > 1e-8
            or np.linalg.norm(projector - projector.conj().T, ord="fro") > 1e-8
            or np.linalg.norm(projector @ projector - projector, ord="fro") > 1e-8
        ):
            raise EvidenceIncomplete(f"mapping {name} isometry/projector identity drift")
    if metric == "principal_singular":
        return float(np.min(np.linalg.svd(ia.conj().T @ ib, compute_uv=False)))
    if metric == "projector_frobenius":
        return float(np.linalg.norm(pa - pb, ord="fro"))
    raise EvidenceIncomplete("unknown exact mapping metric")


def materialize_gate(archive: EvidenceArchive, gate: Mapping[str, Any]) -> Material | tuple[Material, Material] | dict[str, Any]:
    gate_id, family, metric, stage = (
        str(gate["gate_id"]), str(gate["family"]), str(gate["metric"]), str(gate["stage"])
    )
    if bool(gate["deterministic"]):
        parts = gate_id.split("/")
        cutoff = int(parts[1][1:])
        anchor = int(archive.config["formal_matrix"]["mapping_anchor_plan_indices"][str(cutoff)])
        cell = archive.plan["cells"][anchor]
        evidence = archive.get(cutoff, str(cell["layer"]), str(cell["pair_group_id"]), str(cell["backend"]))
        def load(role: str) -> np.ndarray:
            return np.asarray(np.load(evidence.roles[role], allow_pickle=False), dtype=np.complex128)
        ia, ib = load("mapping_isometry_a_npy"), load("mapping_isometry_b_npy")
        pa, pb = load("mapping_projector_a_npy"), load("mapping_projector_b_npy")
        if (
            ia.shape != (cutoff, 2) or ib.shape != (cutoff, 2)
            or pa.shape != (cutoff, cutoff)
            or pb.shape != (cutoff, cutoff)
        ):
            raise EvidenceIncomplete(
                f"mapping arrays do not match gate cutoff c{cutoff}"
            )
        point = evaluate_mapping_arrays(metric, ia, ib, pa, pb)
        passed = point >= float(gate["margin"]) if gate["direction"] == "lower" else point <= float(gate["margin"])
        return {"gate_id": gate_id, "point": point, "passed": bool(passed), "stochastic": False}
    if metric == "logical_ptm_entry":
        return _ptm_material(archive, gate)
    if family == "cutoff_mapping":
        parts = gate_id.split("/")
        lower, upper = map(int, parts[1].split("-"))
        if parts[2] == "fault":
            pair, state = f"fault/{parts[3]}", None
            if metric == "cutoff_density_trace_distance":
                return tuple(_density_material(
                    archive, gate, cutoff_a=lower, cutoff_b=upper, layer="fault",
                    pair=pair, backend_a=backend, backend_b=backend, state=state,
                    stage="trajectory", leg=backend,
                ) for backend in ("A", "B"))  # type: ignore[return-value]
            field = "logical_survival"
        elif parts[2] == "logical":
            pair, state, field = f"logical/{parts[3]}/{parts[4]}", None, "logical_survival"
        else:
            pair, state, field = f"shared/{parts[3]}/{parts[4]}", None, "density"
        if field == "density":
            return tuple(_density_material(
                archive, gate, cutoff_a=lower, cutoff_b=upper, layer="shared",
                pair=pair, backend_a=backend, backend_b=backend, state=None,
                stage="round", leg=backend,
            ) for backend in ("A", "B"))  # type: ignore[return-value]
        legs: list[Material] = []
        layer = "fault" if pair.startswith("fault/") else "logical"
        for backend in ("A", "B"):
            lo, hi = archive.get(lower, layer, pair, backend), archive.get(upper, layer, pair, backend)
            il, roots = _indices(lo, stage="trajectory" if layer == "fault" else "round")
            ih, _ = _indices(hi, stage="trajectory" if layer == "fault" else "round")
            values = _numeric(lo, field, il) - _numeric(hi, field, ih)
            if layer == "logical":
                action = pair.split("/")[-1]
                roots = _logical_block_roots(action, len(roots))
            legs.append(array_material(
                gate_id=gate_id, family=family, direction=str(gate["direction"]),
                margin=float(gate["margin"]), kind="scalar_abs_mean", roots=roots,
                values=values, leg=backend,
            ))
        return legs[0], legs[1]
    if family.startswith("shared_"):
        # All bounded-repair shared gates use vacuum_f expected RESET.
        parts = gate_id.split("/")
        backend = parts[-1] if parts[-1] in {"A", "B"} else None
        pair = "shared/vacuum_f/RESET"
        if family == "shared_absolute_tail":
            if backend is None or "c44" not in parts:
                raise EvidenceIncomplete("shared tail gate identity drift")
            cell = archive.get(44, "shared", pair, backend)
            idx, roots = _indices(cell)
            row_q = _numeric(
                cell, "density_quantization_trace_distance_bound", idx
            )
            values = _tail_values(cell, idx, metric, row_q)
            q = float(np.mean(row_q)) * (
                44.0 if metric == "commutator_defect" else 1.0
            )
            return array_material(
                gate_id=gate_id, family=family,
                direction=str(gate["direction"]), margin=float(gate["margin"]),
                kind="scalar_nonnegative_mean", roots=roots, values=values,
                quantization_radius=q,
            )
        increment = next((part for part in parts if "->" in part), None)
        if "ab" in parts:
            position = parts.index("ab")
            if position + 1 >= len(parts) or not parts[position + 1].startswith("c"):
                raise EvidenceIncomplete("shared A/B cutoff identity drift")
            cutoff = int(parts[position + 1][1:])
            if metric == "density_trace_distance":
                return _density_material(archive, gate, cutoff_a=cutoff, cutoff_b=cutoff, layer="shared", pair=pair, backend_a="A", backend_b="B", state=None, stage="round")
            if metric == "logical_survival":
                left = archive.get(cutoff, "shared", pair, "A")
                right = archive.get(cutoff, "shared", pair, "B")
                idx, roots = _indices(left)
                values = (
                    _density_survival(archive, left, idx)
                    - _density_survival(archive, right, idx)
                )
                q = float(np.mean(
                    _numeric(left, "density_quantization_trace_distance_bound", idx)
                    + _numeric(right, "density_quantization_trace_distance_bound", idx)
                ))
            else:
                roots, values, q = _pair_values(archive, cutoff=cutoff, layer="shared", pair=pair, metric=metric, stage="round")
        else:
            if increment is None:
                raise EvidenceIncomplete("shared cutoff increment missing")
            lower, upper = map(int, increment.split("->"))
            if backend is None:
                raise EvidenceIncomplete("shared cutoff backend missing")
            if metric == "density_trace_distance":
                return _density_material(archive, gate, cutoff_a=lower, cutoff_b=upper, layer="shared", pair=pair, backend_a=backend, backend_b=backend, state=None, stage="round")
            lo, hi = archive.get(lower, "shared", pair, backend), archive.get(upper, "shared", pair, backend)
            idx, roots = _indices(lo)
            if metric == "logical_survival":
                values = (
                    _density_survival(archive, lo, idx)
                    - _density_survival(archive, hi, idx)
                )
                q = float(np.mean(
                    _numeric(lo, "density_quantization_trace_distance_bound", idx)
                    + _numeric(hi, "density_quantization_trace_distance_bound", idx)
                ))
            else:
                fields = (
                    ("level_g", "level_e", "level_f")
                    if metric == "level_probability_l1" else (metric,)
                )
                values = np.stack(
                    [_numeric(lo, f, idx) - _numeric(hi, f, idx) for f in fields],
                    axis=1,
                )
                if len(fields) == 1:
                    values = values[:, 0]
                q = 0.0
        return array_material(
            gate_id=gate_id, family=family, direction=str(gate["direction"]),
            margin=float(gate["margin"]), kind=_kind(metric), roots=roots,
            values=values, quantization_radius=q,
        )
    if family.startswith("fault_") and not family == "fault_trajectory_tail":
        parts = gate_id.split("/")
        # IDs always contain c44 tail or lower->upper bounded-repair comparison.
        scenario = next(value for value in ("step", "telegraph", "burst", "compound") if value in parts)
        state = next(value for value in LABELS if value in parts)
        backend = next(value for value in ("A", "B") if value in parts)
        pair = f"fault/{scenario}"
        if family in {"fault_density"}:
            increment = next(value for value in parts if "->" in value)
            lower, upper = map(int, increment.split("->"))
            return _density_material(archive, gate, cutoff_a=lower, cutoff_b=upper, layer="fault", pair=pair, backend_a=backend, backend_b=backend, state=state, stage="terminal")
        if family == "fault_scalar":
            increment = next(value for value in parts if "->" in value)
            lower, upper = map(int, increment.split("->"))
            lo, hi = archive.get(lower, "fault", pair, backend), archive.get(upper, "fault", pair, backend)
            il, roots = _indices(lo, state=state, stage=stage)
            ih, _ = _indices(hi, state=state, stage=stage)
            fields = ("level_g", "level_e", "level_f") if metric == "level_probability_l1" else (metric,)
            values = np.stack([_numeric(lo, f, il) - _numeric(hi, f, ih) for f in fields], axis=1)
            values = _stage_reduce(values, len(roots))
            if len(fields) == 1:
                values = values[:, 0]
            return array_material(gate_id=gate_id, family=family, direction=str(gate["direction"]), margin=float(gate["margin"]), kind=_kind(metric), roots=roots, values=values)
        # Absolute tail gates are nonnegative observed values, never synthetic.
        cell = archive.get(44, "fault", pair, backend)
        idx, roots = _indices(cell, state=state, stage="terminal")
        state_offset = LABELS.index(state) * 768
        row_q = _numeric(
            cell, "density_quantization_trace_distance_bound", idx
        )
        values = _tail_values(
            cell, np.arange(state_offset, state_offset + 768), metric, row_q
        )
        q = float(np.mean(row_q))
        if metric == "commutator_defect":
            q *= 44.0
        return array_material(
            gate_id=gate_id, family=family, direction=str(gate["direction"]),
            margin=float(gate["margin"]), kind="scalar_nonnegative_mean",
            roots=roots, values=values, quantization_radius=q,
        )
    # Old seven-family A/B gates.
    scope = gate_id
    suffix = "/" + metric
    if not scope.endswith(suffix):
        raise EvidenceIncomplete(f"gate scope/metric mismatch {gate_id}")
    scope = scope[:-len(suffix)]
    parts = scope.split("/")
    cutoff = int(parts[1][1:])
    if parts[2] == "probe":
        layer, pair = "probe", f"probe/{parts[3]}"
    elif parts[2] == "logical":
        layer, pair = "logical", f"logical/{parts[3]}/{parts[4]}"
    elif parts[2] == "reset":
        layer, pair = "shared", f"shared/{parts[3]}/RESET"
    elif parts[2] == "fault":
        layer, pair = "fault", f"fault/{parts[3]}"
    else:
        layer, pair = "shared", f"shared/{parts[3]}/{parts[4]}"
    effective_stage = "trajectory" if layer == "fault" else "round"
    if "density" in metric:
        return _density_material(archive, gate, cutoff_a=cutoff, cutoff_b=cutoff, layer=layer, pair=pair, backend_a="A", backend_b="B", state=None, stage=effective_stage)
    roots, values, q = _pair_values(archive, cutoff=cutoff, layer=layer, pair=pair, metric=metric, stage=effective_stage)
    if layer == "logical":
        action = pair.split("/")[-1]
        roots = _logical_block_roots(action, len(roots))
    return array_material(
        gate_id=gate_id, family=family, direction=str(gate["direction"]),
        margin=float(gate["margin"]), kind=_kind(metric), roots=roots,
        values=values, quantization_radius=q,
    )


def verify(
    root: Path, config_path: Path, output_path: Path, *,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    config = _strict_json(config_path)
    if config.get("task_id") != TASK_ID:
        raise EvidenceIncomplete("config task mismatch")
    paths = config["artifact_paths"]
    plan = _strict_json(_safe_inside(root, paths["plan"], "plan"))
    inventory = _strict_json(_safe_inside(root, paths["inventory"], "inventory"))
    _verify_effective_blueprint(root, config)
    for name, binding in config.get("parent_evidence", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
            raise EvidenceIncomplete(f"parent binding schema drift: {name}")
        parent_path = _safe_inside(root, binding["path"], f"parent {name}")
        size, digest = _file_sha(parent_path)
        if size != binding["bytes"] or digest != binding["sha256"]:
            raise EvidenceIncomplete(f"parent binding raw drift: {name}")
    fresh_binding = config["parent_evidence"]["fresh_twin_parent_config"]
    fresh = _strict_json(_safe_inside(root, fresh_binding["path"], "fresh parent"))
    fault_specs = fresh.get("formal_matrix", {}).get("fault_scenarios")
    if not isinstance(fault_specs, dict) or set(fault_specs) != set(config["formal_matrix"]["fault_scenarios"]):
        raise EvidenceIncomplete("frozen parent fault scenario specs drift")
    seal, manifest, config_sha = _verify_release_chain(
        root, config_path, config, plan, inventory
    )
    receipts, raw_audit = audit_transaction(
        root, config, plan, inventory, allow_test_fixture=allow_test_fixture,
        config_sha256=config_sha,
        sealed_source_snapshot_sha256=str(seal["source_snapshot_sha256"]),
        fault_specs=fault_specs,
        physics=fresh["common_physics"],
    )
    if raw_audit["run_id"] != manifest["run_id"]:
        raise EvidenceIncomplete("receipt run ID differs from manifest")
    blueprint = _strict_json(_safe_inside(root, config["effective_blueprint_binding"]["path"], "blueprint"))
    gates = blueprint.get("gates")
    if (
        not isinstance(gates, list) or len(gates) != 3043
        or len({gate.get("gate_id") for gate in gates}) != 3043
        or sum(bool(gate.get("deterministic")) for gate in gates) != 6
    ):
        raise EvidenceIncomplete("effective blueprint closed gate set drift")
    archive = EvidenceArchive(
        root, config, plan, receipts, analysis_config=fresh
    )
    exact: list[dict[str, Any]] = []
    for gate in (row for row in gates if row["deterministic"]):
        value = materialize_gate(archive, gate)
        if not isinstance(value, dict) or value.get("gate_id") != gate["gate_id"]:
            raise EvidenceIncomplete("exact gate identity drift")
        exact.append(value)

    def stochastic_materials(
    ) -> Iterator[Material | tuple[Material, Material]]:
        for gate in (row for row in gates if not row["deterministic"]):
            value = materialize_gate(archive, gate)
            if isinstance(value, dict):
                raise EvidenceIncomplete("stochastic gate became exact")
            legs = value if isinstance(value, tuple) else (value,)
            for material in legs:
                if (
                    material.gate_id != gate["gate_id"]
                    or material.family != gate["family"]
                    or material.direction != gate["direction"]
                    or material.margin != float(gate["margin"])
                    or len(material.roots) != int(gate["cluster_count"])
                ):
                    raise EvidenceIncomplete(
                        f"material/blueprint contract drift: {gate['gate_id']}"
                    )
            yield value

    expected_ids = [str(gate["gate_id"]) for gate in gates if not gate["deterministic"]]
    stats = joint_maxt(
        stochastic_materials(), expected_gate_ids=expected_ids,
        seed=int(config["seed_registry"]["joint_maxt_rademacher"]["start"]),
    )
    pair_groups = {str(cell["pair_group_id"]) for cell in plan["cells"]}
    if len(pair_groups) != 97:
        raise EvidenceIncomplete("frozen raw pair/root group count drift")
    internal_pass = stats["all_stochastic_gates_passed"] and all(row["passed"] for row in exact)
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "raw_audit": raw_audit,
        "effective_blueprint_sha256": config["effective_blueprint_binding"]["sha256"],
        "gate_count": 3043,
        "stochastic_gate_count": 3037,
        "exact_gate_count": 6,
        "exact_gates": exact,
        "joint_maxt": stats,
        "raw_pair_root_group_count": 97,
        "statistical_composite_root_group_count": stats["root_group_count"],
        "predictive_cdf_numerics": {
            "predictive_cdf_estimand": "fixed_513_grid_sup_proxy_not_continuous_ks",
            "grid_point_count": 513,
            "method": "pure_numpy_abramowitz_stegun_7_1_26",
            "per_side_max_absolute_error_bound": 8e-8,
            "ab_sup_external_radius": 2e-7,
            "radius_included_outside_multiplier": True,
            "continuous_supremum_claim": None,
        },
        "internal_twin_equivalence_verdict": "PASS" if internal_pass else "NO_GO",
        "qualified_claim": None,
        "claim_boundary": {field: None for field in CLAIM_FIELDS},
        "official_or_puviani_surpass": None,
        "postselection_used": False,
        "synthetic_influence_used": False,
        "cross_state_averaging_used": False,
    }
    report["analysis_sha256"] = _sha(report)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(report) + b"\n"
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise EvidenceIncomplete("conflicting immutable verifier report")
    else:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, delete=False, prefix=".verify-", suffix=".tmp"
        ) as handle:
            temp = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, output_path)
        except FileExistsError:
            if output_path.read_bytes() != payload:
                raise EvidenceIncomplete("verifier publication race conflict")
        finally:
            temp.unlink(missing_ok=True)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/phase9/t_risk_20260728_04_powered_twin_qualification.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = _strict_json((args.root / args.config).resolve())
    registered = config.get("artifact_paths", {}).get("independent_verification")
    if args.output is None:
        if not isinstance(registered, str) or not registered:
            raise EvidenceIncomplete(
                "config artifact_paths.independent_verification is not frozen"
            )
        output = args.root / registered
    else:
        output = args.root / args.output
        if isinstance(registered, str) and output.resolve() != (args.root / registered).resolve():
            raise EvidenceIncomplete("CLI output differs from frozen verifier path")
    verify(args.root, (args.root / args.config).resolve(), output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
