"""Independent state/stage diagnostic for the high-cutoff design pilot."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import phase9_high_cutoff_design_pilot as pilot_runner
from cnn_fpga.benchmark.phase9_paired_cluster_uq import (
    NormUCB,
    paired_density_trace_ucb,
    paired_vector_norm_ucb,
)


TASK_ID = "T-RISK-20260727-01"
CONFIG_PATH = "configs/phase9/t_risk_20260727_01_high_cutoff_design_pilot_fresh2.json"
UQ_REPORT_PATH = "docs/t_risk_20260727_01_uq_calibration.json"
UQ_EXTENSION_PATH = "docs/t_risk_20260727_01_uq_power_extension.json"
REPORT_PATH = "docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh2.json"
SOURCE_PATH = (
    "docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh2_source_data.csv"
)
SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-DIAGNOSTIC-V2"
STATUS = "HIGH_CUTOFF_STATE_STAGE_DESIGN_DIAGNOSTIC_COMPLETE"
UQ_CLAIM_BOUNDARY = {
    "calibration_only": True,
    "external_sota": None,
    "hardware_measured": None,
    "ler": None,
    "lifetime": None,
    "official_puviani_exact": None,
    "physical_break_even": None,
    "puviani_nmf_surpass": None,
    "twin_qualification": None,
}
EXTENSION_CLAIM_BOUNDARY = {
    "external_sota": None,
    "hardware_measured": None,
    "ler": None,
    "lifetime": None,
    "official_puviani_exact": None,
    "physical_break_even": None,
    "power_extension_only": True,
    "puviani_nmf_surpass": None,
    "twin_qualification": None,
}


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


def _verify_self_hash(payload: Mapping[str, Any], label: str) -> None:
    unsigned = dict(payload)
    analysis = unsigned.pop("analysis_sha256", None)
    if not isinstance(analysis, str) or analysis != _sha(unsigned):
        raise ValueError(f"{label} self-hash drift")


def _verify_live_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} binding schema drift")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} binding escapes root") from exc
    if dict(binding) != _binding(path, root):
        raise ValueError(f"{label} live binding drift")
    return path


def _verify_report_bindings(root: Path, report: Mapping[str, Any], label: str) -> None:
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError(f"{label} bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"{label}/{name} binding type drift")
        _verify_live_binding(root, binding, f"{label}/{name}")


def _seed(namespace: int, gate_id: str) -> int:
    return (namespace << 64) | int.from_bytes(
        sha256(gate_id.encode("utf-8")).digest()[:8], "big"
    )


def _embed_density(matrix: np.ndarray, lower: int, upper: int) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (3 * lower, 3 * lower) or not 0 < lower < upper:
        raise ValueError("cutoff density embedding mismatch")
    output = np.zeros((3 * upper, 3 * upper), dtype=np.complex128)
    output[: 3 * lower, : 3 * lower] = value
    return output


def _load_inputs(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    config, base = pilot_runner.load_pilot_config(root, require_hardened=True)
    if pilot_runner.CONFIG_PATH != CONFIG_PATH:
        raise ValueError("diagnostic/pilot config path drift")
    execution = pilot_runner.materialize_execution_config(config, base)
    cells = pilot_runner.build_pilot_cells(config, execution)
    run_identity_path = root / str(config["artifact_paths"]["run_identity"])
    run_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
    _verify_self_hash(run_identity, "pilot run identity")
    manifest_path = root / str(config["artifact_paths"]["execution_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_self_hash(manifest, "pilot manifest")
    pilot_runner._verify_manifest(
        root, config, execution, cells, run_identity, manifest
    )
    if (
        manifest.get("task_id") != TASK_ID
        or manifest.get("schema_version") != pilot_runner.MANIFEST_SCHEMA
        or manifest.get("status") != pilot_runner.STATUS
        or manifest.get("scientific_verdict") is not None
        or manifest.get("qualified_claim") is not None
        or manifest.get("exception_rows") != 0
        or manifest.get("conservation_failure_rows") != 0
        or manifest.get("observed_cells") != 32
        or manifest.get("observed_rows") != 27648
        or manifest.get("claim_state") != pilot_runner.CLAIM_BOUNDARY
    ):
        raise ValueError("high-cutoff pilot manifest incomplete or contaminated")
    uq = json.loads((root / UQ_REPORT_PATH).read_text(encoding="utf-8"))
    _verify_self_hash(uq, "UQ calibration report")
    _verify_report_bindings(root, uq, "UQ calibration report")
    required = config["diagnostic_contract"]["calibration_factor_source"]
    if (
        uq.get("analysis_sha256") != required["required_analysis_sha256"]
        or uq.get("claim_state") != UQ_CLAIM_BOUNDARY
        or uq.get("selected_calibration_factor") != required["required_factor"]
        or uq.get("validation_coverage_summary", {}).get("all_cells_passed")
        is not required["required_coverage_all_passed"]
    ):
        raise ValueError("coverage-calibrated factor binding drift")
    extension = json.loads((root / UQ_EXTENSION_PATH).read_text(encoding="utf-8"))
    _verify_self_hash(extension, "UQ power extension report")
    _verify_report_bindings(root, extension, "UQ power extension report")
    if (
        extension.get("verdict") != "PASS_PAIRED_CLUSTER_UQ_POWER_EXTENSION"
        or extension.get("claim_state") != EXTENSION_CLAIM_BOUNDARY
        or extension.get("selected_formal_clusters_per_state") is None
        or extension.get("parent_analysis_sha256") != uq["analysis_sha256"]
    ):
        raise ValueError("UQ power extension is not qualified")
    return config, manifest, uq, extension


def _parse_chunk(
    root: Path,
    receipt: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    _verify_self_hash(receipt, "pilot receipt")
    cell_payload = receipt.get("cell")
    if not isinstance(cell_payload, Mapping):
        raise ValueError("pilot receipt cell missing")
    chunk_id = str(cell_payload.get("chunk_id"))
    expected_receipt_path = (
        root / str(config["artifact_paths"]["receipt_directory"]) / f"{chunk_id}.json"
    )
    live_receipt_path = _verify_live_binding(
        root, receipt_binding, "pilot receipt file"
    )
    if live_receipt_path != expected_receipt_path.resolve():
        raise ValueError("pilot receipt path identity drift")
    live_receipt = json.loads(live_receipt_path.read_text(encoding="utf-8"))
    if live_receipt != receipt:
        raise ValueError("pilot manifest/live receipt drift")
    if (
        receipt.get("task_id") != TASK_ID
        or receipt.get("schema_version") != pilot_runner.RECEIPT_SCHEMA
        or receipt.get("run_id") != manifest.get("run_id")
        or receipt.get("run_identity_analysis_sha256")
        != manifest.get("run_identity_analysis_sha256")
        or receipt.get("config_analysis_sha256")
        != manifest.get("config_analysis_sha256")
        or receipt.get("execution_analysis_sha256")
        != manifest.get("execution_analysis_sha256")
        or receipt.get("pilot_source_sha256") != manifest.get("pilot_source_sha256")
        or receipt.get("chunk_id") != chunk_id
    ):
        raise ValueError("pilot receipt identity drift")
    csv_binding = receipt["csv"]
    npz_binding = receipt["npz"]
    if _binding(root / str(csv_binding["path"]), root) != dict(csv_binding):
        raise ValueError("pilot CSV binding drift")
    if _binding(root / str(npz_binding["path"]), root) != dict(npz_binding):
        raise ValueError("pilot NPZ binding drift")
    rows: list[dict[str, Any]] = []
    with (root / str(csv_binding["path"])).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for raw in csv.DictReader(stream):
            if raw["exception_type"]:
                raise ValueError("pilot contains exception row")
            if raw["conservation_pass"] != "True":
                raise ValueError("pilot contains conservation failure")
            rows.append(
                {
                    **raw,
                    "cutoff": int(raw["cutoff"]),
                    "seed_position": int(raw["seed_position"]),
                    "round_index": int(raw["round_index"]),
                    "terminal_round": raw["terminal_round"].lower() == "true",
                    "mean_photon": float(raw["mean_photon"]),
                    "level_g": float(raw["level_g"]),
                    "level_e": float(raw["level_e"]),
                    "level_f": float(raw["level_f"]),
                    "logical_survival": float(raw["logical_survival"]),
                    "density_quantization_trace_distance_bound": float(
                        raw["density_quantization_trace_distance_bound"]
                    ),
                }
            )
    with np.load(root / str(npz_binding["path"]), allow_pickle=False) as archive:
        density_ids = [str(value) for value in archive["density_row_ids"].tolist()]
        densities = np.asarray(archive["densities"], dtype=np.complex128)
    if len(density_ids) != len(densities):
        raise ValueError("pilot density row alignment drift")
    if len(densities):
        hermitian_error = float(
            np.max(np.abs(densities - densities.conj().transpose(0, 2, 1)))
        )
        trace_error = float(np.max(np.abs(np.trace(densities, axis1=1, axis2=2) - 1.0)))
        minimum_eigenvalue = min(
            float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min())
            for matrix in densities
        )
        if hermitian_error > 5e-5 or trace_error > 5e-5 or minimum_eigenvalue < -5e-5:
            raise ValueError("pilot density physicality drift")
    return rows, dict(zip(density_ids, densities))


def load_pilot_evidence(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    densities: dict[str, np.ndarray] = {}
    receipts = manifest["chunk_receipts"]
    receipt_bindings = manifest["receipt_bindings"]
    if len(receipts) != len(receipt_bindings):
        raise ValueError("pilot receipt binding denominator drift")
    for receipt, receipt_binding in zip(receipts, receipt_bindings):
        chunk_rows, chunk_densities = _parse_chunk(
            root,
            receipt,
            receipt_binding,
            config=config,
            manifest=manifest,
        )
        rows.extend(chunk_rows)
        if set(densities) & set(chunk_densities):
            raise ValueError("duplicate pilot density row id")
        densities.update(chunk_densities)
    if len(rows) != manifest["observed_rows"]:
        raise ValueError("pilot diagnostic row denominator drift")
    terminal_ids = {str(row["row_id"]) for row in rows if row["terminal_round"]}
    if set(densities) != terminal_ids:
        raise ValueError("pilot terminal density coverage drift")
    return rows, densities


def _indexed_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["cutoff"]),
            str(row["scenario"]),
            str(row["backend"]),
            str(row["logical_label"]),
            int(row["seed_position"]),
        )
        grouped.setdefault(key, []).append(dict(row))
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["round_index"]))
        expected_label = ("0", "1", "+", "-", "+i", "-i")[key[4] % 6]
        if key[3] != expected_label or [row["round_index"] for row in values] != list(
            range(12)
        ):
            raise ValueError("pilot state schedule or round coverage drift")
    return grouped


def _state_positions(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
) -> list[int]:
    values = sorted(
        key[4] for key in grouped if key[:4] == (cutoff, scenario, backend, state)
    )
    if len(values) != 12 or any(
        position % 6 != ("0", "1", "+", "-", "+i", "-i").index(state)
        for position in values
    ):
        raise ValueError("pilot per-state cluster denominator drift")
    return values


def _stage_matrix(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    *,
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
    rounds: Sequence[int],
    fields: Sequence[str],
) -> tuple[np.ndarray, list[int]]:
    positions = _state_positions(grouped, cutoff, scenario, backend, state)
    matrix = []
    selected = set(int(value) for value in rounds)
    for position in positions:
        values = [
            row
            for row in grouped[(cutoff, scenario, backend, state, position)]
            if int(row["round_index"]) in selected
        ]
        if len(values) != len(selected):
            raise ValueError("pilot stage round denominator drift")
        matrix.append(
            [float(np.mean([float(row[field]) for row in values])) for field in fields]
        )
    return np.asarray(matrix, dtype=np.float64), positions


def _terminal_density_stack(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    densities: Mapping[str, np.ndarray],
    *,
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    positions = _state_positions(grouped, cutoff, scenario, backend, state)
    stack = []
    quantization = []
    for position in positions:
        terminal = grouped[(cutoff, scenario, backend, state, position)][-1]
        if not terminal["terminal_round"]:
            raise ValueError("pilot terminal row drift")
        stack.append(densities[str(terminal["row_id"])])
        quantization.append(
            float(terminal["density_quantization_trace_distance_bound"])
        )
    return (
        np.asarray(stack, dtype=np.complex128),
        np.asarray(quantization, dtype=np.float64),
        positions,
    )


def _result_row(
    *,
    gate_id: str,
    contrast: str,
    scenario: str,
    state: str,
    stage: str,
    metric: str,
    margin: float,
    ucb: NormUCB,
    cutoff: str,
    backend: str,
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "contrast": contrast,
        "scenario": scenario,
        "logical_state": state,
        "stage": stage,
        "metric": metric,
        "cutoff_or_increment": cutoff,
        "backend_or_pair": backend,
        "estimate": ucb.estimate,
        "raw_radius": ucb.raw_radius,
        "calibrated_radius": ucb.calibrated_radius,
        "quantization_bound": ucb.quantization_bound,
        "upper_bound": ucb.upper_bound,
        "margin": margin,
        "pilot_pass": ucb.upper_bound <= margin,
        "cluster_count": ucb.cluster_count,
        "multiplier_replicates": ucb.multiplier_replicates,
        "multiplier_seed": ucb.seed,
        "design_pilot_only": True,
    }


def evaluate_diagnostics(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    densities: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    grouped = _indexed_rows(rows)
    contract = config["diagnostic_contract"]
    factor = float(contract["calibration_factor_source"]["required_factor"])
    confidence = float(contract["confidence"])
    replicates = int(contract["multiplier_replicates"])
    namespace = int(contract["multiplier_seed_namespace"])
    margins = contract["margins"]
    states = list(config["logical_state_schedule"])
    results: list[dict[str, object]] = []

    def vector_result(
        *,
        gate_id: str,
        left: np.ndarray,
        right: np.ndarray,
        ord_value: float,
        margin_key: str,
        contrast: str,
        scenario: str,
        state: str,
        stage: str,
        metric: str,
        cutoff: str,
        backend: str,
    ) -> None:
        ucb = paired_vector_norm_ucb(
            left,
            right,
            ord_value=ord_value,
            confidence=confidence,
            multiplier_replicates=replicates,
            seed=_seed(namespace, gate_id),
            calibration_factor=factor,
        )
        results.append(
            _result_row(
                gate_id=gate_id,
                contrast=contrast,
                scenario=scenario,
                state=state,
                stage=stage,
                metric=metric,
                margin=float(margins[margin_key]),
                ucb=ucb,
                cutoff=cutoff,
                backend=backend,
            )
        )

    for cutoff in config["cutoffs"]:
        for scenario in config["scenario_names"]:
            for state in states:
                density_a, quant_a, positions_a = _terminal_density_stack(
                    grouped,
                    densities,
                    cutoff=int(cutoff),
                    scenario=scenario,
                    backend="A",
                    state=state,
                )
                density_b, quant_b, positions_b = _terminal_density_stack(
                    grouped,
                    densities,
                    cutoff=int(cutoff),
                    scenario=scenario,
                    backend="B",
                    state=state,
                )
                if positions_a != positions_b:
                    raise ValueError("pilot A/B state positions drift")
                gate_id = f"ab/c{cutoff}/{scenario}/{state}/terminal_density"
                density_ucb = paired_density_trace_ucb(
                    density_a,
                    density_b,
                    confidence=confidence,
                    multiplier_replicates=replicates,
                    seed=_seed(namespace, gate_id),
                    calibration_factor=factor,
                    quantization_bounds=quant_a + quant_b,
                )
                results.append(
                    _result_row(
                        gate_id=gate_id,
                        contrast="same_cutoff_ab",
                        scenario=scenario,
                        state=state,
                        stage="terminal",
                        metric="density_trace_distance",
                        margin=float(margins["ab_terminal_density_trace_distance"]),
                        ucb=density_ucb,
                        cutoff=str(cutoff),
                        backend="A/B",
                    )
                )
                for stage, stage_rounds in config["stage_partition"][scenario].items():
                    a_values, a_positions = _stage_matrix(
                        grouped,
                        cutoff=int(cutoff),
                        scenario=scenario,
                        backend="A",
                        state=state,
                        rounds=stage_rounds,
                        fields=(
                            "mean_photon",
                            "level_g",
                            "level_e",
                            "level_f",
                            "logical_survival",
                        ),
                    )
                    b_values, b_positions = _stage_matrix(
                        grouped,
                        cutoff=int(cutoff),
                        scenario=scenario,
                        backend="B",
                        state=state,
                        rounds=stage_rounds,
                        fields=(
                            "mean_photon",
                            "level_g",
                            "level_e",
                            "level_f",
                            "logical_survival",
                        ),
                    )
                    if a_positions != b_positions:
                        raise ValueError("pilot A/B stage positions drift")
                    specs = (
                        (
                            "mean_photon",
                            a_values[:, 0],
                            b_values[:, 0],
                            1,
                            "ab_terminal_mean_photon_difference",
                        ),
                        (
                            "level_probability_l1",
                            a_values[:, 1:4],
                            b_values[:, 1:4],
                            1,
                            "ab_terminal_level_probability_l1",
                        ),
                        (
                            "logical_survival",
                            a_values[:, 4],
                            b_values[:, 4],
                            1,
                            "ab_terminal_logical_survival_difference",
                        ),
                    )
                    for metric, left, right, ord_value, margin_key in specs:
                        stage_gate = f"ab/c{cutoff}/{scenario}/{state}/{stage}/{metric}"
                        vector_result(
                            gate_id=stage_gate,
                            left=left,
                            right=right,
                            ord_value=ord_value,
                            margin_key=margin_key,
                            contrast="same_cutoff_ab",
                            scenario=scenario,
                            state=state,
                            stage=stage,
                            metric=metric,
                            cutoff=str(cutoff),
                            backend="A/B",
                        )

    for lower, upper in zip(config["cutoffs"][:-1], config["cutoffs"][1:]):
        for scenario in config["scenario_names"]:
            for state in states:
                for backend in ("A", "B"):
                    low_density, low_quant, low_positions = _terminal_density_stack(
                        grouped,
                        densities,
                        cutoff=int(lower),
                        scenario=scenario,
                        backend=backend,
                        state=state,
                    )
                    high_density, high_quant, high_positions = _terminal_density_stack(
                        grouped,
                        densities,
                        cutoff=int(upper),
                        scenario=scenario,
                        backend=backend,
                        state=state,
                    )
                    if low_positions != high_positions:
                        raise ValueError("pilot cutoff state positions drift")
                    embedded = np.asarray(
                        [
                            _embed_density(value, int(lower), int(upper))
                            for value in low_density
                        ]
                    )
                    gate_id = (
                        f"cutoff/{lower}-{upper}/{backend}/{scenario}/{state}/"
                        "terminal_density"
                    )
                    density_ucb = paired_density_trace_ucb(
                        embedded,
                        high_density,
                        confidence=confidence,
                        multiplier_replicates=replicates,
                        seed=_seed(namespace, gate_id),
                        calibration_factor=factor,
                        quantization_bounds=low_quant + high_quant,
                    )
                    results.append(
                        _result_row(
                            gate_id=gate_id,
                            contrast="within_backend_cutoff",
                            scenario=scenario,
                            state=state,
                            stage="terminal",
                            metric="density_trace_distance",
                            margin=float(
                                margins["cutoff_terminal_density_trace_distance"]
                            ),
                            ucb=density_ucb,
                            cutoff=f"{lower}->{upper}",
                            backend=backend,
                        )
                    )
                    for stage, stage_rounds in config["stage_partition"][
                        scenario
                    ].items():
                        low_values, low_stage_positions = _stage_matrix(
                            grouped,
                            cutoff=int(lower),
                            scenario=scenario,
                            backend=backend,
                            state=state,
                            rounds=stage_rounds,
                            fields=(
                                "mean_photon",
                                "level_g",
                                "level_e",
                                "level_f",
                                "logical_survival",
                            ),
                        )
                        high_values, high_stage_positions = _stage_matrix(
                            grouped,
                            cutoff=int(upper),
                            scenario=scenario,
                            backend=backend,
                            state=state,
                            rounds=stage_rounds,
                            fields=(
                                "mean_photon",
                                "level_g",
                                "level_e",
                                "level_f",
                                "logical_survival",
                            ),
                        )
                        if low_stage_positions != high_stage_positions:
                            raise ValueError("pilot cutoff stage positions drift")
                        specs = (
                            (
                                "mean_photon",
                                low_values[:, 0],
                                high_values[:, 0],
                                1,
                                "cutoff_terminal_mean_photon_difference",
                            ),
                            (
                                "level_probability_l1",
                                low_values[:, 1:4],
                                high_values[:, 1:4],
                                1,
                                "cutoff_terminal_level_probability_l1",
                            ),
                            (
                                "logical_survival",
                                low_values[:, 4],
                                high_values[:, 4],
                                1,
                                "cutoff_terminal_logical_survival_difference",
                            ),
                        )
                        for metric, left, right, ord_value, margin_key in specs:
                            stage_gate = (
                                f"cutoff/{lower}-{upper}/{backend}/{scenario}/"
                                f"{state}/{stage}/{metric}"
                            )
                            vector_result(
                                gate_id=stage_gate,
                                left=left,
                                right=right,
                                ord_value=ord_value,
                                margin_key=margin_key,
                                contrast="within_backend_cutoff",
                                scenario=scenario,
                                state=state,
                                stage=stage,
                                metric=metric,
                                cutoff=f"{lower}->{upper}",
                                backend=backend,
                            )
    identifiers = [str(row["gate_id"]) for row in results]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate pilot diagnostic gate id")
    return results


def _build_report_core(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    config, manifest, uq, extension = _load_inputs(root)
    rows, densities = load_pilot_evidence(root, config, manifest)
    diagnostics = evaluate_diagnostics(config, rows, densities)
    cutoff_24_28 = [
        row
        for row in diagnostics
        if row["contrast"] == "within_backend_cutoff"
        and row["metric"] == "density_trace_distance"
        and row["cutoff_or_increment"] == "24->28"
    ]
    trigger_threshold = 0.075
    trigger_32 = any(float(row["estimate"]) > trigger_threshold for row in cutoff_24_28)
    aggregate_ledger: list[dict[str, Any]] = []
    for contrast, metric in sorted(
        {(str(row["contrast"]), str(row["metric"])) for row in diagnostics}
    ):
        strata = [
            row
            for row in diagnostics
            if row["contrast"] == contrast and row["metric"] == metric
        ]
        worst = max(strata, key=lambda row: float(row["upper_bound"]))
        failed = [str(row["gate_id"]) for row in strata if not bool(row["pilot_pass"])]
        aggregate_ledger.append(
            {
                "contrast": contrast,
                "metric": metric,
                "stratum_count": len(strata),
                "failed_stratum_count": len(failed),
                "failed_gate_ids": failed,
                "worst_gate_id": str(worst["gate_id"]),
                "worst_upper_bound": float(worst["upper_bound"]),
                "global_iut_pass": not failed,
                "design_pilot_only": True,
            }
        )
    global_failed = [
        str(row["gate_id"]) for row in diagnostics if not bool(row["pilot_pass"])
    ]
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "status": STATUS,
        "scientific_verdict": None,
        "qualified_claim": None,
        "design_pilot_only": True,
        "diagnostic_count": len(diagnostics),
        "pilot_pass_count": sum(bool(row["pilot_pass"]) for row in diagnostics),
        "pilot_fail_count": sum(not bool(row["pilot_pass"]) for row in diagnostics),
        "design_iut": {
            "aggregation": "state×scenario×backend IUT/max",
            "stratum_count": len(diagnostics),
            "failed_stratum_count": len(global_failed),
            "failed_gate_ids": global_failed,
            "global_iut_pass": not global_failed,
            "aggregate_ledger": aggregate_ledger,
            "formal_qualification_effect": None,
        },
        "maxima": {
            metric: {
                "point_estimate": max(
                    float(row["estimate"])
                    for row in diagnostics
                    if row["metric"] == metric
                ),
                "upper_bound": max(
                    float(row["upper_bound"])
                    for row in diagnostics
                    if row["metric"] == metric
                ),
            }
            for metric in sorted({str(row["metric"]) for row in diagnostics})
        },
        "cutoff_32_decision": {
            "registered_trigger": (
                "any state×scenario×backend 24->28 terminal-density point > 0.075"
            ),
            "threshold": trigger_threshold,
            "observed_maximum_point": max(
                float(row["estimate"]) for row in cutoff_24_28
            ),
            "triggered": trigger_32,
        },
        "formal_design": {
            "calibration_factor": uq["selected_calibration_factor"],
            "clusters_per_state": extension["selected_formal_clusters_per_state"],
            "design_multiplier_replicates": 199,
            "formal_1999_replicate_recalibration_required": True,
            "aggregation": "state×scenario×backend IUT/max",
            "pilot_rows_never_promoted": True,
        },
        "claim_state": dict(config["claim_boundary"]),
        "bindings": {
            "config": _binding(root / CONFIG_PATH, root),
            "pilot_manifest": _binding(
                root / str(config["artifact_paths"]["execution_manifest"]), root
            ),
            "uq_calibration": _binding(root / UQ_REPORT_PATH, root),
            "uq_power_extension": _binding(root / UQ_EXTENSION_PATH, root),
            "uq_hardened_confirmation": _binding(
                root / str(config["hardened_confirmation_source"]["report"]["path"]),
                root,
            ),
            "uq_hardened_confirmation_source_data": _binding(
                root
                / str(config["hardened_confirmation_source"]["source_data"]["path"]),
                root,
            ),
            "diagnostic_source": _binding(Path(__file__).resolve(), root),
            "paired_cluster_uq_source": _binding(
                root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py", root
            ),
        },
    }
    return report, diagnostics


def build_report(root: Path) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Rebuild a finalized report and require its committed source-data file."""

    report, diagnostics = _build_report_core(root)
    source_path = root / SOURCE_PATH
    if not source_path.is_file():
        raise ValueError("diagnostic source data is absent")
    report["bindings"]["source_data"] = _binding(source_path, root)
    report["transaction"] = {
        "write_order": ["source_data", "report"],
        "source_committed_before_report": True,
    }
    report["analysis_sha256"] = _sha(report)
    return report, diagnostics


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


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report, rows = _build_report_core(base)
    fields = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        _atomic_text(base / SOURCE_PATH, stream.read())
    report["bindings"]["source_data"] = _binding(base / SOURCE_PATH, base)
    report["transaction"] = {
        "write_order": ["source_data", "report"],
        "source_committed_before_report": True,
    }
    report["analysis_sha256"] = _sha(report)
    _atomic_text(
        base / REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    rebuilt, rebuilt_rows = build_report(base)
    if rebuilt != report or rebuilt_rows != rows:
        raise RuntimeError("finalized diagnostic live validation drift")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the high-cutoff state/stage design pilot."
    )
    parser.parse_args(argv)
    report = write_artifacts()
    print(
        json.dumps(
            {
                "status": report["status"],
                "analysis_sha256": report["analysis_sha256"],
                "diagnostic_count": report["diagnostic_count"],
                "cutoff_32_triggered": report["cutoff_32_decision"]["triggered"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_PATH",
    "SOURCE_PATH",
    "STATUS",
    "build_report",
    "evaluate_diagnostics",
    "load_pilot_evidence",
    "write_artifacts",
]
