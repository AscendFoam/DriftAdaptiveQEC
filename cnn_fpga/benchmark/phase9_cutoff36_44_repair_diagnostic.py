"""Point-screen diagnostic for the bounded cutoff-36/40/44 repair.

The raw writer never emits a scientific verdict.  This reader reopens all 30
receipt/CSV/NPZ triples and all six Rao--Blackwell sidecars, derives shared
logical survival from the archived expected density, and evaluates the frozen
1,454-gate contract:

* both 36->40 and 40->44 increments;
* state-conditioned cutoff-44 absolute tails;
* same-cutoff A/B expected RESET density/levels;
* no prior raw row or passing gate.

Any failed gate yields the terminal design verdict
``TERMINAL_NO_GO_CUTOFF44_REPAIR``.  An all-gate pass merely permits a new,
separately frozen powered formal; it does not release one here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from cnn_fpga.benchmark import phase9_cutoff32_36_design_diagnostic_v2 as legacy
from cnn_fpga.benchmark import phase9_cutoff36_44_repair as raw_writer


TASK_ID = "T-RISK-20260728-03"
SCHEMA = "PHASE9-CUTOFF36-44-BOUNDED-REPAIR-DIAGNOSTIC-V1"
STATUS = "CUTOFF36_44_BOUNDED_REPAIR_DIAGNOSTIC_COMPLETE"
PASS_VERDICT = "DESIGN_REPAIR_PASS_MAY_PREREGISTER_SEPARATE_POWERED_FORMAL"
NO_GO_VERDICT = "TERMINAL_NO_GO_CUTOFF44_REPAIR"
CLAIM_BOUNDARY = dict(raw_writer.CLAIM_BOUNDARY)
SOURCE_FIELDS = legacy.SOURCE_FIELDS
_SOURCE_SHA256_AT_IMPORT = sha256(Path(__file__).read_bytes()).hexdigest()


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


@contextmanager
def _exclusive_lock(path: Path) -> Iterable[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = str(uuid4())
    document = {"task_id": TASK_ID, "pid": os.getpid(), "token": token}
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
    except FileExistsError as exc:
        raise RuntimeError("repair diagnostic owner lock exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(document))
            stream.flush()
            os.fsync(stream.fileno())
        yield document
    finally:
        if path.exists():
            live = json.loads(path.read_bytes())
            if live != document:
                raise RuntimeError("repair diagnostic owner-lock drift")
            path.unlink()


def _fault_index(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[tuple[int, str, str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, str, str, int], list[dict[str, Any]]] = {}
    schedule = tuple(config["logical_state_schedule"])
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
        for cutoff in config["cutoffs"]
        for scenario in config["scenario_names"]
        for backend in ("A", "B")
    }
    if {key[:3] for key in grouped} != expected_cells:
        raise RuntimeError("repair fault cell coverage drift")
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["round_index"]))
        if key[3] != schedule[key[4] % len(schedule)] or [
            row["round_index"] for row in values
        ] != list(range(12)):
            raise RuntimeError("repair fault state/round coverage drift")
    if len(grouped) != 3 * 4 * 2 * 72:
        raise RuntimeError("repair fault trajectory denominator drift")
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
        raise RuntimeError("repair fault per-state denominator drift")
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
    fields = (
        "mean_photon",
        "level_g",
        "level_e",
        "level_f",
        "logical_survival",
    )
    selected = set(rounds)
    matrix = []
    for position in _fault_positions(grouped, cutoff, scenario, backend, state):
        values = [
            row
            for row in grouped[(cutoff, scenario, backend, state, position)]
            if row["round_index"] in selected
        ]
        if len(values) != len(selected):
            raise RuntimeError("repair fault stage denominator drift")
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
            raise RuntimeError("repair fault terminal row drift")
        stack.append(densities[str(row["row_id"])])
        quantization.append(float(row["density_quantization_trace_distance_bound"]))
    return (
        np.asarray(stack, dtype=np.complex128),
        np.asarray(quantization, dtype=np.float64),
    )


def _shared_index(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
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
            or row["convergence_role"] != "rao_blackwell_expected_reset_repair"
        ):
            raise RuntimeError("expected RESET row identity drift")
        grouped.setdefault((int(row["cutoff"]), str(row["backend"])), []).append(
            dict(row)
        )
    expected = {
        (cutoff, backend) for cutoff in config["cutoffs"] for backend in ("A", "B")
    }
    if set(grouped) != expected:
        raise RuntimeError("expected RESET cell coverage drift")
    for values in grouped.values():
        values.sort(key=lambda row: int(row["seed_position"]))
        if [row["seed_position"] for row in values] != list(range(72)):
            raise RuntimeError("expected RESET denominator drift")
    return grouped


def _populate_logical_survival(
    config: Mapping[str, Any],
    execution: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    densities: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    projectors: dict[tuple[int, str], np.ndarray] = {}
    projector_errors: dict[str, float] = {}
    for cutoff in config["cutoffs"]:
        built = raw_writer.runner.build_simulators(execution, cutoff)
        for backend in ("A", "B"):
            simulator = built[backend]
            isometry = np.asarray(
                (
                    simulator._logical_engine().code_basis.isometry
                    if backend == "A"
                    else simulator._comb_isometry()
                ),
                dtype=np.complex128,
            )
            if isometry.shape != (cutoff, 2):
                raise RuntimeError("repair logical isometry shape drift")
            error = float(
                np.linalg.norm(
                    isometry.conj().T @ isometry - np.eye(2),
                    ord="fro",
                )
            )
            if not np.isfinite(error) or error > 1.0e-10:
                raise RuntimeError("repair logical isometry drift")
            projectors[(cutoff, backend)] = isometry @ isometry.conj().T
            projector_errors[f"c{cutoff}/{backend}"] = error
    fault_count = 0
    shared_count = 0
    maximum_delta = 0.0
    maximum_allowance = 0.0
    shared_values = []
    for row in rows:
        if not row["terminal_round"]:
            continue
        density = np.asarray(densities[str(row["row_id"])])
        cutoff = int(row["cutoff"])
        backend = str(row["backend"])
        oscillator = np.trace(
            density.reshape(cutoff, 3, cutoff, 3),
            axis1=1,
            axis2=3,
        )
        oscillator = (oscillator + oscillator.conj().T) / 2.0
        value = float(np.trace(projectors[(cutoff, backend)] @ oscillator).real)
        if not np.isfinite(value) or not -5e-6 <= value <= 1.0 + 5e-6:
            raise RuntimeError("density-derived logical survival invalid")
        value = float(np.clip(value, 0.0, 1.0))
        if row["layer"] == "fault":
            recorded = row["logical_survival"]
            certificate = row["density_quantization_trace_distance_bound"]
            if recorded is None or certificate is None:
                raise RuntimeError("fault logical-survival evidence missing")
            allowance = float(certificate) + 1.0e-6
            delta = abs(float(recorded) - value)
            if delta > allowance:
                raise RuntimeError("fault logical-survival cross-check failed")
            maximum_delta = max(maximum_delta, delta)
            maximum_allowance = max(maximum_allowance, allowance)
            fault_count += 1
        elif row["layer"] == "shared":
            if row["logical_survival"] is not None:
                raise RuntimeError("shared logical survival was not blank")
            row["logical_survival"] = value
            shared_values.append(value)
            shared_count += 1
        else:
            raise RuntimeError("repair logical projection layer drift")
    if fault_count != 1728 or shared_count != 432:
        raise RuntimeError("repair logical projection denominator drift")
    return {
        "schema_version": "PHASE9-CUTOFF36-44-LOGICAL-PROJECTION-AUDIT-V1",
        "source": "archived_primary_density_and_native_backend_isometry",
        "fault_terminal_cross_checks": fault_count,
        "shared_expected_density_derivations": shared_count,
        "maximum_fault_absolute_delta": maximum_delta,
        "maximum_fault_allowed_delta": maximum_allowance,
        "shared_minimum": min(shared_values),
        "shared_maximum": max(shared_values),
        "projector_orthonormality_frobenius": projector_errors,
    }


def load_evidence(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    list[dict[str, object]],
    dict[str, Any],
]:
    config, base = raw_writer.load_config(root)
    owner = root / str(config["artifact_paths"]["owner_lock"])
    if owner.exists():
        raise RuntimeError("raw repair supervisor active or stale")
    raw_writer._activate_verified_modules(root, config)
    execution = raw_writer.materialize_execution(config, base)
    cells = raw_writer.build_cells(config, execution)
    identity = json.loads(
        (root / str(config["artifact_paths"]["run_identity"])).read_bytes()
    )
    _self_hash(identity, "repair run identity")
    manifest_path = root / str(config["artifact_paths"]["execution_manifest"])
    manifest = json.loads(manifest_path.read_bytes())
    raw_writer._verify_manifest(
        root,
        config,
        execution,
        cells,
        identity,
        identity["input_snapshot_analysis_sha256"],
        manifest,
    )
    heartbeat = json.loads(
        (root / str(config["artifact_paths"]["heartbeat"])).read_bytes()
    )
    if (
        heartbeat.get("state") != "COMPLETE"
        or heartbeat.get("active") is not False
        or heartbeat.get("completed_cells") != 30
        or heartbeat.get("manifest_analysis_sha256") != manifest["analysis_sha256"]
    ):
        raise RuntimeError("raw repair completion marker drift")
    rows: list[dict[str, Any]] = []
    densities: dict[str, np.ndarray] = {}
    raw_bindings: list[dict[str, object]] = []
    for receipt, receipt_binding in zip(
        manifest["chunk_receipts"],
        manifest["receipt_bindings"],
        strict=True,
    ):
        chunk_rows, chunk_densities = legacy._parse_receipt(
            root,
            receipt,
            receipt_binding,
        )
        rows.extend(chunk_rows)
        if set(densities) & set(chunk_densities):
            raise RuntimeError("duplicate repair density row ID")
        densities.update(chunk_densities)
        raw_bindings.extend(
            [
                dict(receipt_binding),
                dict(receipt["csv"]),
                dict(receipt["npz"]),
            ]
        )
        if receipt["cell"]["layer"] == "shared":
            raw_bindings.append(dict(receipt["rao_blackwell_sidecar"]))
    if (
        len(rows) != 21168
        or len(densities) != 2160
        or len({str(row["row_id"]) for row in rows}) != 21168
        or len(raw_bindings) != 96
    ):
        raise RuntimeError("repair raw denominator/binding drift")
    audit = _populate_logical_survival(
        config,
        execution,
        rows,
        densities,
    )
    return config, manifest, rows, densities, raw_bindings, audit


def evaluate(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    densities: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    contract = config["diagnostic_contract"]
    margins = contract["margins"]
    grouped = _fault_index(rows, config)
    output: list[dict[str, object]] = []
    states = tuple(config["logical_state_schedule"])
    for lower, upper in config["required_consecutive_increments"]:
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
                        legacy._gate(
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
                            estimate=legacy._density_point(
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
                        for metric, estimate in legacy._scalar_points(
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
                                legacy._gate(
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
    terminal_cutoff = int(config["absolute_tail_cutoff"])
    for scenario in config["scenario_names"]:
        for backend in ("A", "B"):
            for state in states:
                stack, quantization = _fault_terminal(
                    grouped,
                    densities,
                    cutoff=terminal_cutoff,
                    scenario=scenario,
                    backend=backend,
                    state=state,
                )
                for metric, values in legacy._tail_features(
                    stack, terminal_cutoff
                ).items():
                    output.append(
                        legacy._gate(
                            gate_id=(
                                f"fault/tail/{metric}/c{terminal_cutoff}/"
                                f"{scenario}/{backend}/{state}"
                            ),
                            family="fault_absolute_tail",
                            contrast=f"absolute_cutoff{terminal_cutoff}_tail",
                            scenario=scenario,
                            state=state,
                            stage="terminal",
                            metric=metric,
                            cutoff=str(terminal_cutoff),
                            backend=backend,
                            estimate=float(np.mean(values)),
                            quantization=legacy._tail_quantization_bound(
                                metric,
                                quantization,
                                terminal_cutoff,
                            ),
                            margin=float(margins[f"absolute_terminal_{metric}"]),
                            clusters=12,
                        )
                    )

    shared = _shared_index(rows, config)

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
                ]
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

    for lower, upper in config["required_consecutive_increments"]:
        for backend in ("A", "B"):
            lower_stack, lower_q = shared_stack(lower, backend)
            upper_stack, upper_q = shared_stack(upper, backend)
            output.append(
                legacy._gate(
                    gate_id=f"shared/density/{lower}->{upper}/{backend}",
                    family="shared_density",
                    contrast="within_backend_cutoff",
                    scenario="shared_vacuum_f_expected_reset",
                    state="not_applicable",
                    stage="terminal",
                    metric="density_trace_distance",
                    cutoff=f"{lower}->{upper}",
                    backend=backend,
                    estimate=legacy._density_point(
                        lower_stack,
                        upper_stack,
                        lower,
                        upper,
                    ),
                    quantization=float(np.mean(lower_q) + np.mean(upper_q)),
                    margin=float(margins["cutoff_terminal_density_trace_distance"]),
                    clusters=72,
                )
            )
            for metric, estimate in legacy._scalar_points(
                shared_scalar(lower, backend),
                shared_scalar(upper, backend),
            ).items():
                margin_key = {
                    "mean_photon": ("cutoff_terminal_mean_photon_difference"),
                    "level_probability_l1": ("cutoff_terminal_level_probability_l1"),
                    "logical_survival": ("cutoff_terminal_logical_survival_difference"),
                }[metric]
                output.append(
                    legacy._gate(
                        gate_id=(
                            f"shared/scalar/{metric}/" f"{lower}->{upper}/{backend}"
                        ),
                        family="shared_scalar",
                        contrast="within_backend_cutoff",
                        scenario="shared_vacuum_f_expected_reset",
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
    for cutoff in config["cutoffs"]:
        stack_a, q_a = shared_stack(cutoff, "A")
        stack_b, q_b = shared_stack(cutoff, "B")
        output.append(
            legacy._gate(
                gate_id=f"shared/density/ab/c{cutoff}",
                family="shared_density",
                contrast="same_cutoff_ab",
                scenario="shared_vacuum_f_expected_reset",
                state="not_applicable",
                stage="terminal",
                metric="density_trace_distance",
                cutoff=str(cutoff),
                backend="A/B",
                estimate=legacy._trace_distance(
                    np.mean(stack_a, axis=0),
                    np.mean(stack_b, axis=0),
                ),
                quantization=float(np.mean(q_a) + np.mean(q_b)),
                margin=float(margins["ab_terminal_density_trace_distance"]),
                clusters=72,
            )
        )
        for metric, estimate in legacy._scalar_points(
            shared_scalar(cutoff, "A"),
            shared_scalar(cutoff, "B"),
        ).items():
            margin_key = {
                "mean_photon": "ab_terminal_mean_photon_difference",
                "level_probability_l1": ("ab_terminal_level_probability_l1"),
                "logical_survival": ("ab_terminal_logical_survival_difference"),
            }[metric]
            output.append(
                legacy._gate(
                    gate_id=f"shared/scalar/{metric}/ab/c{cutoff}",
                    family="shared_scalar",
                    contrast="same_cutoff_ab",
                    scenario="shared_vacuum_f_expected_reset",
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
        stack, quantization = shared_stack(terminal_cutoff, backend)
        for metric, values in legacy._tail_features(stack, terminal_cutoff).items():
            output.append(
                legacy._gate(
                    gate_id=(f"shared/tail/{metric}/c{terminal_cutoff}/{backend}"),
                    family="shared_absolute_tail",
                    contrast=f"absolute_cutoff{terminal_cutoff}_tail",
                    scenario="shared_vacuum_f_expected_reset",
                    state="not_applicable",
                    stage="terminal",
                    metric=metric,
                    cutoff=str(terminal_cutoff),
                    backend=backend,
                    estimate=float(np.mean(values)),
                    quantization=legacy._tail_quantization_bound(
                        metric,
                        quantization,
                        terminal_cutoff,
                    ),
                    margin=float(margins[f"absolute_terminal_{metric}"]),
                    clusters=72,
                )
            )
    if len(output) != contract["expected_gate_count"]:
        raise RuntimeError("repair diagnostic gate denominator drift")
    ids = [str(row["gate_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise RuntimeError("repair diagnostic gate ID collision")
    observed: dict[str, int] = {}
    for row in output:
        family = str(row["family"])
        observed[family] = observed.get(family, 0) + 1
    if observed != contract["gate_accounting"]:
        raise RuntimeError("repair diagnostic family accounting drift")
    return sorted(output, key=lambda row: str(row["gate_id"]))


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=SOURCE_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in SOURCE_FIELDS})
    return buffer.getvalue().encode("utf-8")


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    repository = (root or _root()).resolve()
    config, manifest, rows, densities, raw_bindings, audit = load_evidence(repository)
    report_path = repository / str(config["artifact_paths"]["diagnostic_report"])
    source_path = repository / str(config["artifact_paths"]["diagnostic_source_data"])
    lock_path = (
        repository
        / str(config["artifact_paths"]["run_directory"])
        / "diagnostic.owner.lock"
    )
    with _exclusive_lock(lock_path):
        if report_path.exists() or source_path.exists():
            if not report_path.exists() or not source_path.exists():
                raise RuntimeError("partial repair diagnostic publication")
            report = json.loads(report_path.read_bytes())
            _self_hash(report, "published repair diagnostic")
            if report["bindings"]["source_data"] != _binding(
                source_path, repository
            ) or report["bindings"]["raw_manifest"] != _binding(
                repository / str(config["artifact_paths"]["execution_manifest"]),
                repository,
            ):
                raise RuntimeError("published repair diagnostic binding drift")
            return report
        gates = evaluate(config, rows, densities)
        source_payload = _csv_bytes(gates)
        _atomic_bytes(source_path, source_payload)
        passed = [row for row in gates if row["passed"]]
        failed = [row for row in gates if not row["passed"]]
        verdict = PASS_VERDICT if not failed else NO_GO_VERDICT
        family_counts: dict[str, dict[str, int]] = {}
        for row in gates:
            ledger = family_counts.setdefault(
                str(row["family"]),
                {"total": 0, "passed": 0, "failed": 0},
            )
            ledger["total"] += 1
            ledger["passed" if row["passed"] else "failed"] += 1
        report: dict[str, Any] = {
            "task_id": TASK_ID,
            "schema_version": SCHEMA,
            "status": STATUS,
            "scientific_verdict": verdict,
            "authorization_effect": (
                "SEPARATE_POWERED_FORMAL_PREREGISTRATION_MAY_BEGIN"
                if verdict == PASS_VERDICT
                else "TERMINAL_NO_GO_NO_CUTOFF_EXTENSION_OR_POWERED_FORMAL"
            ),
            "terminal_cutoff": 44,
            "automatic_cutoff_extension_beyond_44": False,
            "old_raw_or_gate_composition": False,
            "powered_formal_released": False,
            "gate_count": len(gates),
            "passed_gate_count": len(passed),
            "failed_gate_count": len(failed),
            "family_counts": family_counts,
            "maximum_margin_ratio": max(
                float(row["conservative_point"]) / float(row["margin"]) for row in gates
            ),
            "failed_gate_ids": [str(row["gate_id"]) for row in failed],
            "gate_rows": gates,
            "raw_binding_count": len(raw_bindings),
            "raw_bindings": raw_bindings,
            "raw_bindings_sha256": _sha(raw_bindings),
            "logical_projection_audit": audit,
            "bindings": {
                "config": _binding(repository / raw_writer.CONFIG_PATH, repository),
                "raw_writer": _binding(Path(raw_writer.__file__).resolve(), repository),
                "legacy_math_reader": _binding(
                    Path(legacy.__file__).resolve(), repository
                ),
                "diagnostic_source": _binding(Path(__file__).resolve(), repository),
                "raw_manifest": _binding(
                    repository / str(config["artifact_paths"]["execution_manifest"]),
                    repository,
                ),
                "source_data": _binding(source_path, repository),
            },
            "qualified_claim": None,
            "claim_state": dict(CLAIM_BOUNDARY),
        }
        report["analysis_sha256"] = _sha(report)
        _atomic_bytes(
            report_path,
            (
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the sealed cutoff36/40/44 repair transaction."
    )
    parser.parse_args(argv)
    report = write_artifacts()
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "scientific_verdict": report["scientific_verdict"],
                "gate_count": report["gate_count"],
                "passed_gate_count": report["passed_gate_count"],
                "failed_gate_count": report["failed_gate_count"],
                "analysis_sha256": report["analysis_sha256"],
                "qualified_claim": report["qualified_claim"],
                "claim_state": report["claim_state"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
