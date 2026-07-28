"""Independent raw-to-gate verifier for T-RISK-20260728-03.

This verifier deliberately does not import the diagnostic writer.  It reopens
the sealed raw manifest, parses every archived row/density, independently
derives shared logical survival, reconstructs all 1,454 estimates and margins,
and compares them with both the Source Data CSV and report JSON.  It never
changes a verdict and never releases a powered formal or a performance claim.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import phase9_cutoff32_36_design_diagnostic_v2 as archive_reader
from cnn_fpga.benchmark import phase9_cutoff36_44_repair as raw_writer


TASK_ID = "T-RISK-20260728-03"
SCHEMA = "PHASE9-CUTOFF36-44-BOUNDED-REPAIR-INDEPENDENT-VERIFIER-V1"
VERIFIED_PASS = "VERIFIED_DESIGN_REPAIR_PASS_MAY_PREREGISTER_SEPARATE_POWERED_FORMAL"
VERIFIED_NO_GO = "VERIFIED_TERMINAL_NO_GO_CUTOFF44_REPAIR"
PASS_VERDICT = "DESIGN_REPAIR_PASS_MAY_PREREGISTER_SEPARATE_POWERED_FORMAL"
NO_GO_VERDICT = "TERMINAL_NO_GO_CUTOFF44_REPAIR"
CLAIM_BOUNDARY = dict(raw_writer.CLAIM_BOUNDARY)
TAIL_METRICS = (
    "top1_fock_mass",
    "top2_fock_mass",
    "top4_fock_mass",
    "normalized_mean_photon",
    "commutator_defect",
)
SCALAR_METRICS = (
    "mean_photon",
    "level_probability_l1",
    "logical_survival",
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


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = left - right
    hermitian = (difference + difference.conj().T) / 2.0
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def _embed(matrix: np.ndarray, lower: int, upper: int) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (3 * lower, 3 * lower) or not 0 < lower < upper:
        raise RuntimeError("independent density embedding drift")
    output = np.zeros((3 * upper, 3 * upper), dtype=np.complex128)
    output[: 3 * lower, : 3 * lower] = value
    return output


def _density_point(
    lower_stack: np.ndarray,
    upper_stack: np.ndarray,
    lower: int,
    upper: int,
) -> float:
    embedded = np.asarray([_embed(matrix, lower, upper) for matrix in lower_stack])
    return _trace_distance(np.mean(embedded, axis=0), np.mean(upper_stack, axis=0))


def _tail_features(stack: np.ndarray, cutoff: int) -> dict[str, np.ndarray]:
    value = np.asarray(stack, dtype=np.complex128)
    if value.ndim != 3 or value.shape[1:] != (3 * cutoff, 3 * cutoff):
        raise RuntimeError("independent tail density shape drift")
    joint = value.reshape(len(value), cutoff, 3, cutoff, 3)
    oscillator = np.trace(joint, axis1=2, axis2=4)
    populations = np.diagonal(oscillator, axis1=1, axis2=2).real
    if (
        not np.all(np.isfinite(populations))
        or float(np.min(populations)) < -5.0e-5
        or float(np.max(np.abs(np.sum(populations, axis=1) - 1.0))) > 5.0e-5
    ):
        raise RuntimeError("independent tail population physicality drift")
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


def _tail_quantization(
    metric: str,
    values: np.ndarray,
    cutoff: int,
) -> float:
    certificates = np.asarray(values, dtype=np.float64)
    if (
        certificates.ndim != 1
        or len(certificates) == 0
        or not np.all(np.isfinite(certificates))
        or np.any(certificates < 0.0)
    ):
        raise RuntimeError("independent tail certificate drift")
    return float(np.mean(certificates)) * (
        float(cutoff) if metric == "commutator_defect" else 1.0
    )


def _scalar_points(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_mean = np.mean(left, axis=0)
    right_mean = np.mean(right, axis=0)
    return {
        "mean_photon": abs(float(left_mean[0] - right_mean[0])),
        "level_probability_l1": float(np.sum(np.abs(left_mean[1:4] - right_mean[1:4]))),
        "logical_survival": abs(float(left_mean[4] - right_mean[4])),
    }


def _load_raw(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    config, base = raw_writer.load_config(root)
    if (root / str(config["artifact_paths"]["owner_lock"])).exists():
        raise RuntimeError("raw repair supervisor active or stale")
    raw_writer._activate_verified_modules(root, config)
    execution = raw_writer.materialize_execution(config, base)
    cells = raw_writer.build_cells(config, execution)
    identity = json.loads(
        (root / str(config["artifact_paths"]["run_identity"])).read_bytes()
    )
    _self_hash(identity, "repair run identity")
    manifest = json.loads(
        (root / str(config["artifact_paths"]["execution_manifest"])).read_bytes()
    )
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
        or heartbeat.get("manifest_analysis_sha256") != manifest["analysis_sha256"]
    ):
        raise RuntimeError("raw repair completion drift")
    rows: list[dict[str, Any]] = []
    densities: dict[str, np.ndarray] = {}
    for receipt, receipt_binding in zip(
        manifest["chunk_receipts"],
        manifest["receipt_bindings"],
        strict=True,
    ):
        chunk_rows, chunk_densities = archive_reader._parse_receipt(
            root,
            receipt,
            receipt_binding,
        )
        rows.extend(chunk_rows)
        if set(densities) & set(chunk_densities):
            raise RuntimeError("independent duplicate density row ID")
        densities.update(chunk_densities)
    if (
        len(rows) != 21168
        or len(densities) != 2160
        or len({str(row["row_id"]) for row in rows}) != 21168
    ):
        raise RuntimeError("independent raw denominator drift")

    projectors: dict[tuple[int, str], np.ndarray] = {}
    for cutoff in config["cutoffs"]:
        simulators = raw_writer.runner.build_simulators(execution, cutoff)
        for backend in ("A", "B"):
            simulator = simulators[backend]
            isometry = np.asarray(
                (
                    simulator._logical_engine().code_basis.isometry
                    if backend == "A"
                    else simulator._comb_isometry()
                )
            )
            error = float(
                np.linalg.norm(
                    isometry.conj().T @ isometry - np.eye(2),
                    ord="fro",
                )
            )
            if isometry.shape != (cutoff, 2) or error > 1.0e-10:
                raise RuntimeError("independent logical isometry drift")
            projectors[(cutoff, backend)] = isometry @ isometry.conj().T
    fault_cross_checks = 0
    shared_derivations = 0
    maximum_fault_delta = 0.0
    for row in rows:
        if not row["terminal_round"]:
            continue
        cutoff = int(row["cutoff"])
        backend = str(row["backend"])
        density = densities[str(row["row_id"])]
        oscillator = np.trace(
            density.reshape(cutoff, 3, cutoff, 3),
            axis1=1,
            axis2=3,
        )
        oscillator = (oscillator + oscillator.conj().T) / 2.0
        survival = float(np.trace(projectors[(cutoff, backend)] @ oscillator).real)
        survival = float(np.clip(survival, 0.0, 1.0))
        if row["layer"] == "fault":
            delta = abs(float(row["logical_survival"]) - survival)
            allowance = float(row["density_quantization_trace_distance_bound"]) + 1.0e-6
            if delta > allowance:
                raise RuntimeError(
                    "independent fault logical-survival cross-check failed"
                )
            maximum_fault_delta = max(maximum_fault_delta, delta)
            fault_cross_checks += 1
        else:
            if row["logical_survival"] is not None:
                raise RuntimeError("independent shared logical-survival blank drift")
            row["logical_survival"] = survival
            shared_derivations += 1
    audit = {
        "fault_terminal_cross_checks": fault_cross_checks,
        "shared_expected_density_derivations": shared_derivations,
        "maximum_fault_absolute_delta": maximum_fault_delta,
    }
    if fault_cross_checks != 1728 or shared_derivations != 432:
        raise RuntimeError("independent logical denominator drift")
    return config, manifest, rows, densities, audit


def _recompute(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    densities: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    fault: dict[tuple[int, str, str, str, int], list[Mapping[str, Any]]] = {}
    shared: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["layer"] == "fault":
            key = (
                int(row["cutoff"]),
                str(row["scenario"]),
                str(row["backend"]),
                str(row["logical_label"]),
                int(row["seed_position"]),
            )
            fault.setdefault(key, []).append(row)
        elif row["layer"] == "shared":
            shared.setdefault((int(row["cutoff"]), str(row["backend"])), []).append(row)
    for values in fault.values():
        values.sort(key=lambda row: int(row["round_index"]))
    for values in shared.values():
        values.sort(key=lambda row: int(row["seed_position"]))
    margins = config["diagnostic_contract"]["margins"]
    output: dict[str, dict[str, Any]] = {}

    def add(
        gate_id: str,
        *,
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
    ) -> None:
        if gate_id in output:
            raise RuntimeError("independent gate ID collision")
        conservative = float(estimate + quantization)
        output[gate_id] = {
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
            "cluster_count": clusters,
            "statistical_role": "UNPOWERED_POINT_SCREEN_NO_EQUIVALENCE_CLAIM",
            "qualification_effect": None,
        }

    def positions(cutoff: int, scenario: str, backend: str, state: str) -> list[int]:
        values = sorted(
            key[4] for key in fault if key[:4] == (cutoff, scenario, backend, state)
        )
        if len(values) != 12:
            raise RuntimeError("independent fault state denominator drift")
        return values

    def terminal(
        cutoff: int, scenario: str, backend: str, state: str
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = [
            fault[(cutoff, scenario, backend, state, position)][-1]
            for position in positions(cutoff, scenario, backend, state)
        ]
        return (
            np.asarray([densities[str(row["row_id"])] for row in selected]),
            np.asarray(
                [
                    float(row["density_quantization_trace_distance_bound"])
                    for row in selected
                ]
            ),
        )

    def stage_values(
        cutoff: int,
        scenario: str,
        backend: str,
        state: str,
        round_indices: Sequence[int],
    ) -> np.ndarray:
        fields = (
            "mean_photon",
            "level_g",
            "level_e",
            "level_f",
            "logical_survival",
        )
        selected_rounds = set(round_indices)
        matrix = []
        for position in positions(cutoff, scenario, backend, state):
            selected = [
                row
                for row in fault[(cutoff, scenario, backend, state, position)]
                if int(row["round_index"]) in selected_rounds
            ]
            matrix.append(
                [
                    float(np.mean([float(row[field]) for row in selected]))
                    for field in fields
                ]
            )
        return np.asarray(matrix)

    states = config["logical_state_schedule"]
    for lower, upper in config["required_consecutive_increments"]:
        for scenario in config["scenario_names"]:
            for backend in ("A", "B"):
                for state in states:
                    low_density, low_q = terminal(lower, scenario, backend, state)
                    high_density, high_q = terminal(upper, scenario, backend, state)
                    add(
                        f"fault/density/{lower}->{upper}/{scenario}/{backend}/{state}",
                        family="fault_density",
                        contrast="within_backend_cutoff",
                        scenario=scenario,
                        state=state,
                        stage="terminal",
                        metric="density_trace_distance",
                        cutoff=f"{lower}->{upper}",
                        backend=backend,
                        estimate=_density_point(
                            low_density,
                            high_density,
                            lower,
                            upper,
                        ),
                        quantization=float(np.mean(low_q) + np.mean(high_q)),
                        margin=float(
                            config["diagnostic_contract"][
                                "design_density_point_threshold"
                            ]
                        ),
                        clusters=12,
                    )
                    for stage, round_indices in config["stage_partition"][
                        scenario
                    ].items():
                        points = _scalar_points(
                            stage_values(
                                lower,
                                scenario,
                                backend,
                                state,
                                round_indices,
                            ),
                            stage_values(
                                upper,
                                scenario,
                                backend,
                                state,
                                round_indices,
                            ),
                        )
                        for metric, estimate in points.items():
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
                            add(
                                (
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
    terminal_cutoff = int(config["absolute_tail_cutoff"])
    for scenario in config["scenario_names"]:
        for backend in ("A", "B"):
            for state in states:
                stack, certificates = terminal(
                    terminal_cutoff, scenario, backend, state
                )
                for metric, values in _tail_features(stack, terminal_cutoff).items():
                    add(
                        (
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
                        quantization=_tail_quantization(
                            metric, certificates, terminal_cutoff
                        ),
                        margin=float(margins[f"absolute_terminal_{metric}"]),
                        clusters=12,
                    )

    def shared_stack(cutoff: int, backend: str) -> tuple[np.ndarray, np.ndarray]:
        values = shared[(cutoff, backend)]
        if len(values) != 72:
            raise RuntimeError("independent shared denominator drift")
        return (
            np.asarray([densities[str(row["row_id"])] for row in values]),
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
            ]
        )

    for lower, upper in config["required_consecutive_increments"]:
        for backend in ("A", "B"):
            low, low_q = shared_stack(lower, backend)
            high, high_q = shared_stack(upper, backend)
            add(
                f"shared/density/{lower}->{upper}/{backend}",
                family="shared_density",
                contrast="within_backend_cutoff",
                scenario="shared_vacuum_f_expected_reset",
                state="not_applicable",
                stage="terminal",
                metric="density_trace_distance",
                cutoff=f"{lower}->{upper}",
                backend=backend,
                estimate=_density_point(low, high, lower, upper),
                quantization=float(np.mean(low_q) + np.mean(high_q)),
                margin=float(margins["cutoff_terminal_density_trace_distance"]),
                clusters=72,
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
                add(
                    f"shared/scalar/{metric}/{lower}->{upper}/{backend}",
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
                        float(np.mean(low_q) + np.mean(high_q))
                        if metric == "logical_survival"
                        else 0.0
                    ),
                    margin=float(margins[margin_key]),
                    clusters=72,
                )
    for cutoff in config["cutoffs"]:
        stack_a, q_a = shared_stack(cutoff, "A")
        stack_b, q_b = shared_stack(cutoff, "B")
        add(
            f"shared/density/ab/c{cutoff}",
            family="shared_density",
            contrast="same_cutoff_ab",
            scenario="shared_vacuum_f_expected_reset",
            state="not_applicable",
            stage="terminal",
            metric="density_trace_distance",
            cutoff=str(cutoff),
            backend="A/B",
            estimate=_trace_distance(
                np.mean(stack_a, axis=0),
                np.mean(stack_b, axis=0),
            ),
            quantization=float(np.mean(q_a) + np.mean(q_b)),
            margin=float(margins["ab_terminal_density_trace_distance"]),
            clusters=72,
        )
        for metric, estimate in _scalar_points(
            shared_scalar(cutoff, "A"),
            shared_scalar(cutoff, "B"),
        ).items():
            margin_key = {
                "mean_photon": "ab_terminal_mean_photon_difference",
                "level_probability_l1": "ab_terminal_level_probability_l1",
                "logical_survival": ("ab_terminal_logical_survival_difference"),
            }[metric]
            add(
                f"shared/scalar/{metric}/ab/c{cutoff}",
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
    for backend in ("A", "B"):
        stack, certificates = shared_stack(terminal_cutoff, backend)
        for metric, values in _tail_features(stack, terminal_cutoff).items():
            add(
                f"shared/tail/{metric}/c{terminal_cutoff}/{backend}",
                family="shared_absolute_tail",
                contrast=f"absolute_cutoff{terminal_cutoff}_tail",
                scenario="shared_vacuum_f_expected_reset",
                state="not_applicable",
                stage="terminal",
                metric=metric,
                cutoff=str(terminal_cutoff),
                backend=backend,
                estimate=float(np.mean(values)),
                quantization=_tail_quantization(metric, certificates, terminal_cutoff),
                margin=float(margins[f"absolute_terminal_{metric}"]),
                clusters=72,
            )
    if len(output) != 1454:
        raise RuntimeError("independent gate denominator drift")
    return output


def _read_source(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != archive_reader.SOURCE_FIELDS:
            raise RuntimeError("independent Source Data schema drift")
        rows = list(reader)
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        gate_id = raw["gate_id"]
        if not gate_id or gate_id in output:
            raise RuntimeError("independent Source Data gate ID drift")
        output[gate_id] = {
            **raw,
            "estimate": float(raw["estimate"]),
            "quantization_bound": float(raw["quantization_bound"]),
            "conservative_point": float(raw["conservative_point"]),
            "margin": float(raw["margin"]),
            "passed": raw["passed"] == "True",
            "cluster_count": int(raw["cluster_count"]),
            "qualification_effect": (
                None
                if raw["qualification_effect"] == ""
                else raw["qualification_effect"]
            ),
        }
    return output


def verify(root: Path | None = None) -> dict[str, Any]:
    repository = (root or _root()).resolve()
    config, manifest, rows, densities, raw_audit = _load_raw(repository)
    report_path = repository / str(config["artifact_paths"]["diagnostic_report"])
    source_path = repository / str(config["artifact_paths"]["diagnostic_source_data"])
    report = json.loads(report_path.read_bytes())
    _self_hash(report, "repair diagnostic report")
    source = _read_source(source_path)
    recomputed = _recompute(config, rows, densities)
    if set(source) != set(recomputed):
        raise RuntimeError("independent gate-set drift")
    numeric_fields = (
        "estimate",
        "quantization_bound",
        "conservative_point",
        "margin",
    )
    exact_fields = (
        "family",
        "contrast",
        "scenario",
        "logical_state",
        "stage",
        "metric",
        "cutoff_or_increment",
        "backend_or_pair",
        "passed",
        "cluster_count",
        "statistical_role",
        "qualification_effect",
    )
    maximum_delta = 0.0
    for gate_id, expected in recomputed.items():
        observed = source[gate_id]
        for field in exact_fields:
            if observed[field] != expected[field]:
                raise RuntimeError(
                    f"independent gate metadata/decision drift: {gate_id}/{field}"
                )
        for field in numeric_fields:
            delta = abs(float(observed[field]) - float(expected[field]))
            maximum_delta = max(maximum_delta, delta)
            if delta > max(2.0e-12, 2.0e-12 * abs(float(expected[field]))):
                raise RuntimeError(
                    f"independent gate arithmetic drift: {gate_id}/{field}"
                )
    report_rows = {str(row["gate_id"]): row for row in report.get("gate_rows", [])}
    if set(report_rows) != set(source):
        raise RuntimeError("report/Source Data gate-set drift")
    for gate_id, source_row in source.items():
        report_row = report_rows[gate_id]
        for field in exact_fields:
            if report_row[field] != source_row[field]:
                raise RuntimeError("report/Source Data metadata drift")
        for field in numeric_fields:
            if abs(float(report_row[field]) - source_row[field]) > 2.0e-12:
                raise RuntimeError("report/Source Data numeric drift")
    failed = sorted(gate_id for gate_id, row in recomputed.items() if not row["passed"])
    verdict = PASS_VERDICT if not failed else NO_GO_VERDICT
    if (
        report.get("task_id") != TASK_ID
        or report.get("gate_count") != 1454
        or report.get("passed_gate_count") != 1454 - len(failed)
        or report.get("failed_gate_count") != len(failed)
        or sorted(report.get("failed_gate_ids", [])) != failed
        or report.get("scientific_verdict") != verdict
        or report.get("automatic_cutoff_extension_beyond_44") is not False
        or report.get("old_raw_or_gate_composition") is not False
        or report.get("powered_formal_released") is not False
        or report.get("qualified_claim") is not None
        or report.get("claim_state") != CLAIM_BOUNDARY
        or report["bindings"]["source_data"] != _binding(source_path, repository)
        or report["bindings"]["raw_manifest"]
        != _binding(
            repository / str(config["artifact_paths"]["execution_manifest"]),
            repository,
        )
    ):
        raise RuntimeError("independent report verdict/claim drift")
    result: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "verification_verdict": (
            VERIFIED_PASS if verdict == PASS_VERDICT else VERIFIED_NO_GO
        ),
        "scientific_verdict": verdict,
        "gate_count": 1454,
        "passed_gate_count": 1454 - len(failed),
        "failed_gate_count": len(failed),
        "failed_gate_ids": failed,
        "maximum_raw_recomputation_absolute_delta": maximum_delta,
        "raw_audit": {
            **raw_audit,
            "row_count": len(rows),
            "density_count": len(densities),
            "manifest_analysis_sha256": manifest["analysis_sha256"],
        },
        "bindings": {
            "config": _binding(repository / raw_writer.CONFIG_PATH, repository),
            "raw_manifest": _binding(
                repository / str(config["artifact_paths"]["execution_manifest"]),
                repository,
            ),
            "diagnostic_report": _binding(report_path, repository),
            "source_data": _binding(source_path, repository),
            "verifier_source": _binding(Path(__file__).resolve(), repository),
            "archive_reader_source": _binding(
                Path(archive_reader.__file__).resolve(), repository
            ),
        },
        "automatic_cutoff_extension_beyond_44": False,
        "old_raw_or_gate_composition": False,
        "powered_formal_released": False,
        "qualified_claim": None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    result["analysis_sha256"] = _sha(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify the cutoff36/40/44 repair."
    )
    parser.parse_args(argv)
    repository = _root()
    result = verify(repository)
    output = repository / str(
        raw_writer.load_config(repository)[0]["artifact_paths"][
            "diagnostic_verification"
        ]
    )
    output.write_bytes(
        (
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    print(
        json.dumps(
            {
                "verification_verdict": result["verification_verdict"],
                "gate_count": result["gate_count"],
                "failed_gate_count": result["failed_gate_count"],
                "analysis_sha256": result["analysis_sha256"],
                "qualified_claim": result["qualified_claim"],
                "claim_state": result["claim_state"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
