"""Fresh T-RISK-20260726-01 dual-backend raw-evidence runner.

This module executes the repaired, powered formal matrix.  It deliberately
does not calculate equivalence gates and never emits a scientific verdict.
The physics-free verifier owns all estimands and PASS/NO-GO/INCOMPLETE
classification.

Execution is a resumable fail-closed transaction:

* seed counts, cells, margins and endpoints are accepted only from the sealed
  config; the CLI exposes no scientific override;
* each cell/backend chunk is committed as an atomic CSV + allow_pickle=False
  NPZ pair and then recorded in an append-only SHA-256 hash chain;
* resume skips a chunk only after exact config/seal/run-id/path/size/hash
  verification;
* every expected row is retained, including explicit exception rows;
* raw IQ is diagnostic/backaction evidence, while predictive IQ moments and
  reset success are Rao--Blackwellized from pre-measurement/pre-reset laws;
* the final raw archive is a self-contained ZIP of independently readable NPZ
  chunks plus a canonical archive manifest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from math import isfinite
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import scipy

from cnn_fpga.benchmark.phase9_dual_backend_qualification import (
    _action_words,
    _apply_intervention,
    _density_diagnostics,
    _fault_delta_for_round,
    _initial_fock_ket,
    _one_step,
)
from physics.phase9_backend_a import (
    BACKEND_A_ID,
    BackendAConfig,
    Phase9BackendASimulator,
)
from physics.phase9_backend_b import BACKEND_B_ID, BackendBConfig
from physics.phase9_backend_b_logical_bridge import (
    MehlerLogicalBridgeConfig,
    Phase9BackendBMehlerBridgeSimulator,
)
from physics.phase9_iq_likelihood_reference import (
    REFERENCE_ID as IQ_REFERENCE_ID,
    evaluate_observation,
    integrated_predictive_moments,
    per_complex_sample_log_score,
)
from physics.phase9_twin_contract import (
    ActionWord,
    NominalAction,
    execute_representative_probe,
    representative_action_probes,
)


TASK_ID = "T-RISK-20260726-01"
CONFIG_SCHEMA = "PHASE9-FRESH-TWIN-QUALIFICATION-CONFIG-V1"
RUNNER_ID = "PHASE9-FRESH-TWIN-RAW-EVIDENCE-RUNNER-V1"
ROW_SCHEMA = "PHASE9-FRESH-TWIN-ROUND-LEDGER-V1"
RAW_ARCHIVE_SCHEMA = "PHASE9-FRESH-TWIN-CHUNKED-RAW-ARCHIVE-V1"
ATTEMPT_SCHEMA = "PHASE9-FRESH-TWIN-ATTEMPT-EVENT-V1"
MANIFEST_SCHEMA = "PHASE9-FRESH-TWIN-EXECUTION-MANIFEST-V1"
HEARTBEAT_SCHEMA = "PHASE9-FRESH-TWIN-RUNNER-HEARTBEAT-V1"
PREFORMAL_SEAL_SCHEMA = "PHASE9-FRESH-TWIN-PREFORMAL-SEAL-V1"
FORMAL_STATUS = "FORMAL_RAW_EVIDENCE_COMPLETE"

CONFIG_PATH = "configs/phase9/t_risk_20260726_01_fresh_twin_qualification.json"

LEDGER_FIELDS = (
    "row_id",
    "row_schema",
    "layer",
    "cell_base",
    "cell_id",
    "backend",
    "backend_id",
    "cutoff",
    "convergence_role",
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
    "pre_readout_i",
    "pre_readout_q",
    "pre_measurement_g",
    "pre_measurement_e",
    "pre_measurement_f",
    "pre_reset_g",
    "pre_reset_e",
    "pre_reset_f",
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
    "predictive_mean_i",
    "predictive_mean_q",
    "predictive_cov_ii",
    "predictive_cov_iq",
    "predictive_cov_qq",
    "heldout_reference_log_evidence",
    "heldout_proper_score_per_sample",
    "heldout_llr_ge_per_sample",
    "heldout_llr_gf_per_sample",
    "heldout_llr_ef_per_sample",
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
)


@dataclass(frozen=True)
class CellSpec:
    """One independently committed cell/backend work unit."""

    chunk_id: str
    layer: str
    cell_base: str
    cutoff: int
    backend: str
    sample_count: int
    convergence_role: str
    action: str = ""
    initial_state: str = ""
    logical_label: str = ""
    probe_id: str = ""
    scenario: str = ""
    horizon: int = 1

    @property
    def expected_rows(self) -> int:
        return self.sample_count * self.horizon


@dataclass
class ChunkEvidence:
    rows: list[dict[str, object]]
    densities: list[np.ndarray]
    density_row_ids: list[str]
    raw_iq: np.ndarray
    heldout_iq: np.ndarray


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _binding(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact escapes repository root: {path}") from exc
    payload = resolved.read_bytes()
    return {"path": relative, "bytes": len(payload), "sha256": _sha_bytes(payload)}


def _binding_for_relative(root: Path, relative: str) -> dict[str, object]:
    return _binding(root / relative, root)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    include_header: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(LEDGER_FIELDS),
            extrasaction="raise",
            lineterminator="\n",
        )
        if include_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _interval(spec: object, name: str) -> tuple[int, int]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{name} must be an object")
    return (
        _strict_int(spec.get("start"), f"{name}.start"),
        _strict_int(spec.get("count"), f"{name}.count", minimum=1),
    )


def _validate_relative_path(relative: object, name: str) -> str:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} escapes the repository")
    return relative


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject any scientific or transaction drift before formal access."""

    if config.get("task_id") != TASK_ID or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("fresh qualification config identity mismatch")
    if config.get("formal_result_accessed_before_freeze") is not False:
        raise ValueError("config was not frozen before formal access")
    historical = config.get("historical_policy")
    if not isinstance(historical, Mapping):
        raise ValueError("historical_policy is missing")
    if (
        historical.get("historical_no_go_rewritten") is not False
        or historical.get("historical_formal_cell_data_access_allowed") is not False
    ):
        raise ValueError("historical NO-GO isolation is not fail closed")
    design = config.get("design_power")
    if not isinstance(design, Mapping):
        raise ValueError("design_power binding is missing")
    if (
        _strict_int(design.get("round_sample_count"), "round_sample_count", minimum=1)
        != 768
        or _strict_int(
            design.get("trajectory_sample_count"),
            "trajectory_sample_count",
            minimum=1,
        )
        != 256
        or design.get("formal_seed_pool_accessed_by_design") is not False
    ):
        raise ValueError("powered formal sample counts drifted")
    matrix = config.get("formal_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("formal_matrix is missing")
    exact = {
        "same_cutoff_ab": [8, 12, 16],
        "cutoff_ladder": [8, 12, 16, 20],
        "primary_cutoff_increments": [[12, 16], [16, 20]],
        "diagnostic_cutoff_increment": [8, 12],
        "tail_actions": ["IDLE", "XZ", "RESET"],
    }
    for key, expected in exact.items():
        if matrix.get(key) != expected:
            raise ValueError(f"formal_matrix.{key} drifted")
    if (
        matrix.get("round_sample_count") != 768
        or matrix.get("trajectory_sample_count") != 256
        or matrix.get("fault_initialization")
        != "logical_six_state_balanced_cycle"
        or matrix.get("fault_logical_label_schedule")
        != ["0", "1", "+", "-", "+i", "-i"]
        or matrix.get("all_cells_required") is not True
        or matrix.get("no_postselection") is not True
        or matrix.get("early_exit_on_scientific_metric_forbidden") is not True
    ):
        raise ValueError("formal matrix count/completeness policy drifted")
    expected_actions = ["IDLE", "X", "Z", "XZ", "RESET", "HOLD", "LKG_HOLD"]
    if matrix.get("nominal_actions") != expected_actions:
        raise ValueError("nominal action set/order drifted")
    probes = matrix.get("representative_probes")
    if not isinstance(probes, list) or len(probes) != 16 or len(set(probes)) != 16:
        raise ValueError("all 16 unique representative probes are required")
    if set(matrix.get("fault_scenarios", {})) != {"step", "telegraph", "burst", "compound"}:
        raise ValueError("fault scenario set drifted")

    splits = config.get("formal_splits")
    if not isinstance(splits, Mapping):
        raise ValueError("formal_splits is missing")
    required_splits = (
        "round_backend_a",
        "round_backend_b",
        "trajectory_backend_a",
        "trajectory_backend_b",
        "heldout_common",
    )
    intervals: list[tuple[int, int, str]] = []
    for name in required_splits:
        start, count = _interval(splits.get(name), f"formal_splits.{name}")
        intervals.append((start, start + count, name))
    for index, (start, stop, name) in enumerate(intervals):
        for other_start, other_stop, other_name in intervals[index + 1 :]:
            if max(start, other_start) < min(stop, other_stop):
                raise ValueError(f"formal seed overlap: {name}/{other_name}")
    if (
        intervals[0][:2] != (1070000, 1070768)
        or intervals[1][:2] != (1071000, 1071768)
        or intervals[2][:2] != (1072000, 1072256)
        or intervals[3][:2] != (1073000, 1073256)
        or intervals[4][:2] != (1076000, 1076768)
    ):
        raise ValueError("fresh formal seed ranges drifted")
    if splits.get("all_intervals_disjoint") is not True:
        raise ValueError("formal split disjointness is not asserted")

    readout = config.get("readout_semantics")
    common = config.get("common_physics")
    if not isinstance(readout, Mapping) or not isinstance(common, Mapping):
        raise ValueError("readout/common physics config missing")
    if (
        common.get("iq_samples") != 8
        or _finite(common.get("iq_sigma"), "iq_sigma") != 0.48
        or readout.get("raw_log_evidence_primary") is not False
        or readout.get("predictive_primary")
        != "Rao-Blackwellized from pre-measurement prior"
    ):
        raise ValueError("fresh IQ semantic repair drifted")
    archive = config.get("archive_policy")
    resume = config.get("resume_policy")
    runner = config.get("runner_policy")
    if (
        not isinstance(archive, Mapping)
        or archive.get("format") != "zip_of_allow_pickle_false_npz_chunks"
        or not isinstance(resume, Mapping)
        or resume.get("nan_to_zero") is not False
        or not isinstance(runner, Mapping)
        or runner.get("runner_emits_scientific_verdict") is not False
        or runner.get("formal_cli_overrides") != []
    ):
        raise ValueError("archive/resume/no-verdict policy drifted")
    paths = config.get("artifact_paths")
    if not isinstance(paths, Mapping):
        raise ValueError("artifact_paths missing")
    normalized = [
        _validate_relative_path(paths.get(name), f"artifact_paths.{name}")
        for name in (
            "attempt_ledger",
            "cell_ledger",
            "raw_archive",
            "execution_manifest",
            "heartbeat",
            "chunk_directory",
        )
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError("artifact paths must be unique")
    dependencies = config.get("runtime_dependencies")
    expected_dependencies = [
        "cnn_fpga/benchmark/phase9_dual_backend_qualification.py",
        "physics/phase9_backend_a.py",
        "physics/phase9_backend_b.py",
        "physics/phase9_backend_b_logical_bridge.py",
        "physics/phase9_iq_likelihood_reference.py",
        "physics/phase9_twin_contract.py",
        "physics/fock_density_model.py",
        "physics/fock_sbs_cycle.py",
        "physics/finite_energy_gkp.py",
        "physics/quadrature_conventions.py",
        "physics/sbs_error_space.py",
    ]
    if (
        not isinstance(dependencies, Mapping)
        or dependencies.get("all_must_be_byte_bound_by_preformal_seal")
        is not True
        or dependencies.get("paths") != expected_dependencies
    ):
        raise ValueError("runtime dependency seal contract drifted")
    environment = config.get("runtime_environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("python") != [3, 12, 7]
        or environment.get("numpy") != "1.26.4"
        or environment.get("scipy") != "1.13.1"
    ):
        raise ValueError("formal numerical runtime contract drifted")
    contract = config.get("verification_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("gate_blueprint_ref") != "#/gate_blueprint/rows"
        or contract.get("blueprint_sha256_ref")
        != "#/gate_blueprint/canonical_blueprint_sha256"
        or contract.get("cluster_unit") != "seed_position"
        or contract.get("global_test") != "intersection_union_equivalence"
        or contract.get("cell_test") != "two_one_sided_tests"
        or contract.get("cell_confidence_interval") != 0.9
        or contract.get("tost_z") != 1.6448536269514722
        or contract.get("raw_log_evidence_primary") is not False
        or contract.get("fault_mixed_unit_composite") is not False
        or contract.get("drift_normalization") != [0.12, 0.1, 0.18, 0.14, 0.12]
        or contract.get("aggregate_rescue_forbidden") is not True
        or contract.get("missing_nonfinite_exception")
        != "INCOMPLETE_FAIL_CLOSED"
        or contract.get("density_quantization_bound_must_be_added") is not True
    ):
        raise ValueError("verification contract drifted")
    claim_boundary = config.get("claim_boundary")
    expected_claim_fields = {
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
    }
    if (
        not isinstance(claim_boundary, Mapping)
        or claim_boundary.get("runner_qualified_claim") is not None
        or not isinstance(claim_boundary.get("current_claim_state"), Mapping)
        or set(claim_boundary["current_claim_state"]) != expected_claim_fields
        or any(
            value is not None
            for value in claim_boundary["current_claim_state"].values()
        )
    ):
        raise ValueError("runner claim boundary must remain literal null")
    blueprint = config.get("gate_blueprint")
    if not isinstance(blueprint, Mapping):
        raise ValueError("materialized gate_blueprint is missing")
    rows = blueprint.get("rows")
    if (
        not isinstance(rows, list)
        or blueprint.get("row_count") != 1589
        or len(rows) != 1589
        or blueprint.get("source_design_blueprint_sha256")
        != "bf586a2f5ba6096ad4446f1c18b30eeff4d0cd13c9ef523ed482dd293d76e24b"
    ):
        raise ValueError("gate blueprint count/source binding drifted")
    required_blueprint_fields = {
        "gate_id",
        "family",
        "stage",
        "metric",
        "margin",
        "normalized_sd",
        "deterministic",
        "direction",
    }
    identifiers: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != required_blueprint_fields:
            raise ValueError(f"gate_blueprint row {index} schema drifted")
        gate_id = row.get("gate_id")
        if not isinstance(gate_id, str) or not gate_id:
            raise ValueError(f"gate_blueprint row {index} lacks gate_id")
        identifiers.append(gate_id)
        _finite(row.get("margin"), f"gate_blueprint[{index}].margin")
        _finite(
            row.get("normalized_sd"),
            f"gate_blueprint[{index}].normalized_sd",
        )
        if not isinstance(row.get("deterministic"), bool):
            raise ValueError("gate_blueprint deterministic must be bool")
        expected_direction = (
            "lower" if row.get("metric") == "principal_singular" else "upper"
        )
        if row.get("direction") != expected_direction:
            raise ValueError("gate_blueprint direction drifted")
    if len(set(identifiers)) != 1589:
        raise ValueError("gate blueprint IDs are not unique")
    blueprint_hash = _sha_bytes(_canonical(rows))
    if (
        blueprint.get("canonical_blueprint_sha256") != blueprint_hash
    ):
        raise ValueError("materialized gate blueprint hash mismatch")


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, object]]:
    path = root / CONFIG_PATH
    config = _strict_json(path)
    validate_config(config)
    return config, _binding(path, root)


def _analysis_sha(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    claimed = unsigned.pop("analysis_sha256", None)
    actual = _sha_bytes(_canonical(unsigned))
    if claimed != actual:
        raise RuntimeError("document analysis_sha256 mismatch")
    return actual


def verify_preformal_seal(
    root: Path,
    config: Mapping[str, Any],
    config_binding: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, object]]:
    """Byte-bind the outcome-blind seal and every required live input."""

    specification = config["preformal_seal"]
    seal_path = root / str(specification["path"])
    seal = _strict_json(seal_path)
    if (
        seal.get("task_id") != TASK_ID
        or seal.get("schema_version") != PREFORMAL_SEAL_SCHEMA
        or seal.get("status") != specification["required_status"]
        or seal.get("formal_result_accessed") is not False
        or seal.get("historical_formal_cell_data_accessed") is not False
        or seal.get("all_gates_passed") is not True
        or seal.get("all_mutations_detected") is not True
        or seal.get("scientific_verdict") is not None
    ):
        raise RuntimeError("fresh preformal seal policy mismatch")
    _analysis_sha(seal)
    bindings = seal.get("live_bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("preformal seal lacks live_bindings")
    by_path: dict[str, Mapping[str, object]] = {}
    for name, binding in bindings.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise RuntimeError("invalid preformal live binding")
        path = binding.get("path")
        if not isinstance(path, str):
            raise RuntimeError("preformal live binding lacks path")
        live = _binding_for_relative(root, path)
        if dict(binding) != live:
            raise RuntimeError(f"preformal live binding drift: {name}")
        by_path[path] = binding
    required_paths = {
        CONFIG_PATH,
        "cnn_fpga/benchmark/phase9_fresh_twin_qualification.py",
        str(config["historical_policy"]["historical_lineage_receipt"]["path"]),
        str(config["design_power"]["path"]),
        str(config["preformal_seal"]["audit_path"]),
        *[str(path) for path in config["runtime_dependencies"]["paths"]],
    }
    if not required_paths.issubset(by_path):
        raise RuntimeError("preformal seal omits a required runner input")
    if dict(by_path[CONFIG_PATH]) != dict(config_binding):
        raise RuntimeError("seal does not bind the loaded config bytes")

    for section, key in (
        ("historical_policy", "historical_lineage_receipt"),
        ("design_power", None),
    ):
        spec = config[section] if key is None else config[section][key]
        document = _strict_json(root / str(spec["path"]))
        _analysis_sha(document)
        if (
            document.get("task_id") != TASK_ID
            or document.get("schema_version") != spec["schema_version"]
            or document.get("verdict") != spec["required_verdict"]
        ):
            raise RuntimeError(f"{section} semantic binding mismatch")
        if section == "design_power" and (
            document.get("blueprint", {}).get("gate_count") != 1589
            or document.get("blueprint", {}).get("sha256")
            != config["gate_blueprint"]["source_design_blueprint_sha256"]
        ):
            raise RuntimeError("design-power/materialized blueprint binding mismatch")
    return seal, _binding(seal_path, root)


def _common_config_kwargs(
    config: Mapping[str, Any],
    *,
    cutoff: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = dict(config["common_physics"])
    segment_steps = _strict_int(common.pop("segment_steps"), "segment_steps", minimum=1)
    common["cutoff"] = cutoff
    common["iq_centers"] = tuple(tuple(float(value) for value in row) for row in common["iq_centers"])
    common["drift_retention"] = tuple(common["drift_retention"])
    common["drift_noise_std"] = tuple(common["drift_noise_std"])
    mappings = config["native_logical_mappings"]
    a_kwargs = dict(common)
    a_kwargs.update(
        {
            "substeps_per_segment": segment_steps,
            "logical_projector_delta": float(mappings["backend_a"]["projector_delta"]),
            "logical_grid_points": int(mappings["backend_a"]["grid_points"]),
        }
    )
    b_kwargs = dict(common)
    b_kwargs.update(
        {
            "split_steps_per_segment": segment_steps,
            "comb_squeezing": float(mappings["backend_b"]["comb_squeezing"]),
            "comb_envelope": float(mappings["backend_b"]["comb_envelope"]),
            "comb_half_width": int(mappings["backend_b"]["comb_half_width"]),
        }
    )
    return a_kwargs, b_kwargs


def build_simulators(config: Mapping[str, Any], cutoff: int) -> dict[str, object]:
    a_kwargs, b_kwargs = _common_config_kwargs(config, cutoff=cutoff)
    bridge = config["native_logical_mappings"]["backend_b_mehler_bridge"]
    return {
        "A": Phase9BackendASimulator(BackendAConfig(**a_kwargs)),
        "B": Phase9BackendBMehlerBridgeSimulator(
            BackendBConfig(**b_kwargs),
            bridge_config=MehlerLogicalBridgeConfig(
                projector_delta=float(bridge["projector_delta"]),
                tail_tolerance=float(bridge["tail_tolerance"]),
            ),
        ),
    }


def _roles_for_cutoff(cutoff: int) -> str:
    roles: list[str] = []
    if cutoff in (8, 12, 16):
        roles.append("same_cutoff_ab")
    if cutoff in (12, 16):
        roles.append("primary_12_to_16")
    if cutoff in (16, 20):
        roles.append("primary_16_to_20")
    if cutoff in (8, 12):
        roles.append("diagnostic_8_to_12")
    return "+".join(roles)


def _safe_chunk_id(value: str) -> str:
    digest = _sha_bytes(value.encode("utf-8"))[:16]
    readable = "".join(character if character.isalnum() else "_" for character in value)
    return f"{readable[:96]}__{digest}"


def build_cell_plan(config: Mapping[str, Any]) -> list[CellSpec]:
    """Expand the exact all-required matrix in deterministic order."""

    validate_config(config)
    matrix = config["formal_matrix"]
    round_count = int(matrix["round_sample_count"])
    trajectory_count = int(matrix["trajectory_sample_count"])
    cells: list[CellSpec] = []

    def add(
        *,
        layer: str,
        cell_base: str,
        cutoff: int,
        backend: str,
        sample_count: int,
        **kwargs: object,
    ) -> None:
        key = f"{layer}|c{cutoff}|{cell_base}|{backend}"
        cells.append(
            CellSpec(
                chunk_id=_safe_chunk_id(key),
                layer=layer,
                cell_base=cell_base,
                cutoff=cutoff,
                backend=backend,
                sample_count=sample_count,
                convergence_role=_roles_for_cutoff(cutoff),
                **kwargs,
            )
        )

    for cutoff in matrix["same_cutoff_ab"]:
        for state in matrix["shared_fock_states"]:
            for action in matrix["nominal_actions"]:
                for backend in ("A", "B"):
                    add(
                        layer="shared",
                        cell_base=f"shared|{state}|{action}",
                        cutoff=cutoff,
                        backend=backend,
                        sample_count=round_count,
                        action=action,
                        initial_state=state,
                    )
        for label in matrix["logical_labels"]:
            for action in matrix["nominal_actions"]:
                for backend in ("A", "B"):
                    add(
                        layer="logical",
                        cell_base=f"logical|{label}|{action}",
                        cutoff=cutoff,
                        backend=backend,
                        sample_count=round_count,
                        action=action,
                        logical_label=label,
                    )
        for scenario, specification in matrix["fault_scenarios"].items():
            for backend in ("A", "B"):
                add(
                    layer="fault",
                    cell_base=f"fault|{scenario}",
                    cutoff=cutoff,
                    backend=backend,
                    sample_count=trajectory_count,
                    scenario=scenario,
                    horizon=int(specification["horizon"]),
                )

    cutoff = int(matrix["probe_cutoff"])
    probes = representative_action_probes()
    expected = list(matrix["representative_probes"])
    if [probe.probe_id for probe in probes] != expected:
        raise RuntimeError("representative probe contract/order drifted")
    for probe in probes:
        terminal = execute_representative_probe(probe)[-1]
        action = NominalAction(terminal.recurrence.action_word.action_code).name
        if action != probe.expected_terminal:
            raise RuntimeError(f"probe terminal mismatch: {probe.probe_id}")
        for backend in ("A", "B"):
            add(
                layer="probe",
                cell_base=f"probe|{probe.probe_id}|{action}",
                cutoff=cutoff,
                backend=backend,
                sample_count=round_count,
                action=action,
                probe_id=probe.probe_id,
                initial_state="vacuum_g",
            )

    cutoff = int(matrix["tail_only_cutoff"])
    for state in matrix["shared_fock_states"]:
        for action in matrix["tail_actions"]:
            for backend in ("A", "B"):
                add(
                    layer="shared",
                    cell_base=f"shared|{state}|{action}",
                    cutoff=cutoff,
                    backend=backend,
                    sample_count=round_count,
                    action=action,
                    initial_state=state,
                )
    for label in matrix["logical_labels"]:
        for action in matrix["tail_actions"]:
            for backend in ("A", "B"):
                add(
                    layer="logical",
                    cell_base=f"logical|{label}|{action}",
                    cutoff=cutoff,
                    backend=backend,
                    sample_count=round_count,
                    action=action,
                    logical_label=label,
                )
    for scenario, specification in matrix["fault_scenarios"].items():
        for backend in ("A", "B"):
            add(
                layer="fault",
                cell_base=f"fault|{scenario}",
                cutoff=cutoff,
                backend=backend,
                sample_count=trajectory_count,
                scenario=scenario,
                horizon=int(specification["horizon"]),
            )
    identifiers = [cell.chunk_id for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("cell plan contains duplicate chunk IDs")
    if len(cells) != 592 or sum(cell.expected_rows for cell in cells) != 528384:
        raise RuntimeError("fresh matrix accounting drifted")
    return cells


def expected_cell_count(config: Mapping[str, Any]) -> int:
    return len(build_cell_plan(config))


def expected_row_count(config: Mapping[str, Any]) -> int:
    return sum(cell.expected_rows for cell in build_cell_plan(config))


def _seed_for(config: Mapping[str, Any], cell: CellSpec, position: int) -> int:
    if not 0 <= position < cell.sample_count:
        raise ValueError("seed position outside frozen cell denominator")
    if cell.layer == "fault":
        key = "trajectory_backend_a" if cell.backend == "A" else "trajectory_backend_b"
    else:
        key = "round_backend_a" if cell.backend == "A" else "round_backend_b"
    start, count = _interval(config["formal_splits"][key], key)
    if position >= count:
        raise ValueError("seed position exceeds frozen split")
    return start + position


def _heldout_window(
    config: Mapping[str, Any],
    *,
    cell_base: str,
    cutoff: int,
    position: int,
    round_index: int,
) -> np.ndarray:
    """Generate one backend-common, address-stable heldout IQ window."""

    start, count = _interval(config["formal_splits"]["heldout_common"], "heldout_common")
    if not 0 <= position < count:
        raise ValueError("heldout seed position outside frozen split")
    # Deliberately omit cutoff: the exact same heldout record is used for
    # A/B and for every member of a cutoff ladder at a fixed
    # cell/seed-position/round.  This removes heldout Monte Carlo noise from
    # cutoff contrasts without pairing the backend-native physical RNG.
    _strict_int(cutoff, "cutoff", minimum=1)
    address = _sha_bytes(cell_base.encode("utf-8"))
    words = [int(address[offset : offset + 8], 16) for offset in range(0, 32, 8)]
    sequence = np.random.SeedSequence([start + position, round_index, *words])
    rng = np.random.default_rng(sequence)
    readout = config["readout_semantics"]
    prior = np.asarray(readout["heldout_component_prior"], dtype=np.float64)
    component = int(rng.choice(3, p=prior))
    centers = np.asarray(readout["heldout_centers"], dtype=np.float64)
    sigma = float(readout["heldout_sigma"])
    count_iq = int(config["common_physics"]["iq_samples"])
    return centers[component] + sigma * rng.standard_normal((count_iq, 2))


def _identity(
    cell: CellSpec,
    *,
    seed: int,
    position: int,
    row_index: int,
    action: str,
    round_index: int = 0,
    terminal_round: bool = True,
    logical_label: str | None = None,
) -> dict[str, object]:
    trajectory_id = (
        f"{cell.cell_base}|c{cell.cutoff}|{cell.backend}|p{position:04d}"
        if cell.layer == "fault"
        else ""
    )
    row_id = (
        f"{trajectory_id}|r{round_index:03d}"
        if trajectory_id
        else f"{cell.layer}|c{cell.cutoff}|{cell.cell_base}|{cell.backend}|p{position:04d}"
    )
    if cell.layer == "shared":
        scope = f"ab/c{cell.cutoff}/shared/{cell.initial_state}/{cell.action}"
    elif cell.layer == "probe":
        scope = f"ab/c{cell.cutoff}/probe/{cell.probe_id}"
    elif cell.layer == "logical":
        scope = f"ab/c{cell.cutoff}/logical/{cell.logical_label}/{cell.action}"
    elif cell.layer == "fault":
        scope = f"ab/c{cell.cutoff}/fault/{cell.scenario}"
    else:
        raise ValueError(f"unsupported layer {cell.layer}")
    return {
        "row_id": row_id,
        "row_schema": ROW_SCHEMA,
        "layer": cell.layer,
        "cell_base": cell.cell_base,
        "cell_id": scope,
        "backend": cell.backend,
        "backend_id": BACKEND_A_ID if cell.backend == "A" else BACKEND_B_ID,
        "cutoff": cell.cutoff,
        "convergence_role": cell.convergence_role,
        "seed": seed,
        "seed_position": position,
        "trajectory_id": trajectory_id,
        "round_index": round_index,
        "terminal_round": terminal_round,
        "action": action,
        "probe_id": cell.probe_id,
        "scenario": cell.scenario,
        "initial_state": cell.initial_state,
        "logical_label": (
            cell.logical_label if logical_label is None else logical_label
        ),
        "rng_namespace": (
            "NUMPY_SEEDSEQUENCE_ADDRESSED"
            if cell.backend == "A"
            else "BLAKE2B_ADDRESS_PYTHON_RANDOM_BOX_MULLER"
        ),
        "archive_chunk": cell.chunk_id,
        "archive_row_index": row_index,
    }


def _empty_row(identity: Mapping[str, object]) -> dict[str, object]:
    row = {field: "" for field in LEDGER_FIELDS}
    row.update(identity)
    row.update(
        {
            "density_index": -1,
            "raw_iq_index": -1,
            "heldout_iq_index": int(identity["archive_row_index"]),
            "reset_requested": False,
            "reset_hidden_success": False,
            "leakage_resident": False,
            "conservation_pass": False,
            "exception_type": "",
            "exception_message": "",
        }
    )
    return row


def _truth_levels(result: object, backend: str) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    truth = result.truth
    if backend == "A":
        return (
            tuple(truth.pre_measurement_level_probabilities),
            tuple(truth.pre_reset_level_probabilities),
            tuple(truth.post_reset_level_probabilities),
        )
    return (
        tuple(truth.pre_measurement_levels),
        tuple(truth.pre_reset_levels),
        tuple(truth.post_reset_levels),
    )


def _result_density_and_logical(
    simulator: object,
    result: object,
    backend: str,
) -> tuple[np.ndarray, float, tuple[float, ...] | None]:
    density = np.asarray(result.state.joint_density, dtype=np.complex128)
    oscillator = simulator.oscillator_density(density)
    oscillator_matrix = oscillator.matrix if hasattr(oscillator, "matrix") else oscillator
    number = np.diag(np.arange(result.state.cutoff, dtype=np.float64))
    mean_photon = float(np.trace(number @ oscillator_matrix).real)
    if result.logical is None:
        return density, mean_photon, None
    if backend == "A":
        survival = float(result.logical.code_survival_probability)
        corrected = np.asarray(result.logical.frame_corrected_logical_density)
    else:
        survival = float(result.logical.code_survival)
        corrected = np.asarray(result.logical.corrected_density)
    block = survival * corrected
    values = (
        survival,
        block[0, 0].real,
        block[0, 0].imag,
        block[0, 1].real,
        block[0, 1].imag,
        block[1, 0].real,
        block[1, 0].imag,
        block[1, 1].real,
        block[1, 1].imag,
    )
    return density, mean_photon, tuple(float(value) for value in values)


def _success_row(
    *,
    config: Mapping[str, Any],
    identity: Mapping[str, object],
    simulator: object,
    result: object,
    action: ActionWord,
    heldout: np.ndarray,
    density_index: int,
) -> tuple[dict[str, object], np.ndarray]:
    backend = str(identity["backend"])
    row = _empty_row(identity)
    observation = result.observation
    pre, pre_reset, post_reset = _truth_levels(result, backend)
    drift_before = tuple(float(value) for value in result.truth.drift_before)
    centers = np.asarray(config["common_physics"]["iq_centers"], dtype=np.float64).copy()
    centers[:, 0] += drift_before[2]
    centers[:, 1] += drift_before[3]
    sigma = float(config["common_physics"]["iq_sigma"])
    raw_receipt = evaluate_observation(
        observation.iq_i,
        observation.iq_q,
        priors=pre,
        centers=centers,
        sigma=sigma,
    )
    heldout_receipt = evaluate_observation(
        heldout[:, 0],
        heldout[:, 1],
        priors=pre,
        centers=centers,
        sigma=sigma,
    )
    predictive_mean, predictive_cov = integrated_predictive_moments(
        priors=pre,
        centers=centers,
        sigma=sigma,
        sample_count=int(config["common_physics"]["iq_samples"]),
    )
    density, mean_photon, logical = _result_density_and_logical(
        simulator, result, backend
    )
    density_quantization_frobenius_error: object = ""
    density_quantization_certified_frobenius_bound: object = ""
    density_quantization_trace_distance_bound: object = ""
    if density_index >= 0:
        restored = density.astype(np.complex64).astype(np.complex128)
        density_quantization_frobenius_error = float(
            np.linalg.norm(density - restored, ord="fro")
        )
        unit_roundoff = 2.0 ** -24
        underflow_half_ulp = 2.0 ** -150
        density_quantization_certified_frobenius_bound = float(
            unit_roundoff
            / (1.0 - unit_roundoff)
            * np.linalg.norm(restored, ord="fro")
            + np.sqrt(2.0 * density.size) * underflow_half_ulp
        )
        if (
            density_quantization_frobenius_error
            > density_quantization_certified_frobenius_bound
        ):
            raise RuntimeError("complex64 quantization exceeded certified bound")
        density_quantization_trace_distance_bound = float(
            0.5
            * np.sqrt(density.shape[0])
            * density_quantization_certified_frobenius_bound
        )
    diagnostics = _density_diagnostics(density)
    posterior = tuple(float(value) for value in observation.posterior_levels)
    posterior_error = abs(sum(posterior) - 1.0)
    level_error = abs(sum(post_reset) - 1.0)
    reference_posterior_error = float(
        np.sum(np.abs(np.asarray(posterior) - np.asarray(raw_receipt.posterior)))
    )
    reference_log_error = abs(
        float(observation.log_evidence_density) - raw_receipt.log_evidence
    )
    conservation = (
        np.all(np.isfinite(density))
        and diagnostics[0] <= 5.0e-8
        and diagnostics[1] <= 5.0e-8
        and diagnostics[2] >= -5.0e-8
        and posterior_error <= 5.0e-8
        and level_error <= 5.0e-8
        and reference_posterior_error <= 5.0e-10
        and reference_log_error <= 5.0e-10
        and abs(float(observation.integrated_i) - raw_receipt.integrated_i)
        <= 2.0e-12
        and abs(float(observation.integrated_q) - raw_receipt.integrated_q)
        <= 2.0e-12
    )
    reset_requested = bool(action.reset_request)
    rb_success: object = ""
    if reset_requested:
        rb_success = (
            pre_reset[0]
            + float(config["common_physics"]["reset_success_e"]) * pre_reset[1]
            + float(config["common_physics"]["reset_success_f"]) * pre_reset[2]
        )
    truth = result.truth
    row.update(
        {
            "density_index": density_index,
            "raw_iq_index": int(identity["archive_row_index"]),
            "heldout_window_sha256": _sha_bytes(
                np.asarray(heldout, dtype="<f8").tobytes(order="C")
            ),
            "pre_readout_i": drift_before[2],
            "pre_readout_q": drift_before[3],
            "pre_measurement_g": pre[0],
            "pre_measurement_e": pre[1],
            "pre_measurement_f": pre[2],
            "pre_reset_g": pre_reset[0],
            "pre_reset_e": pre_reset[1],
            "pre_reset_f": pre_reset[2],
            "integrated_i": observation.integrated_i,
            "integrated_q": observation.integrated_q,
            "integrated_i_mean_error": abs(
                float(observation.integrated_i) - raw_receipt.integrated_i
            ),
            "integrated_q_mean_error": abs(
                float(observation.integrated_q) - raw_receipt.integrated_q
            ),
            "raw_log_evidence": observation.log_evidence_density,
            "raw_reference_log_evidence": raw_receipt.log_evidence,
            "raw_within_window_residual": raw_receipt.within_window_residual,
            "posterior_g": posterior[0],
            "posterior_e": posterior[1],
            "posterior_f": posterior[2],
            "level_g": post_reset[0],
            "level_e": post_reset[1],
            "level_f": post_reset[2],
            "mean_photon": mean_photon,
            "reset_requested": reset_requested,
            "reset_hidden_success": truth.reset_hidden_outcome == "success",
            "reset_ack": observation.reset_ack,
            "rao_blackwell_reset_success": rb_success,
            "leakage_resident": result.state.leakage_age > 0,
            "leakage_residence_probability": float(post_reset[2]),
            "leakage_age": result.state.leakage_age,
            "predictive_mean_i": predictive_mean[0],
            "predictive_mean_q": predictive_mean[1],
            "predictive_cov_ii": predictive_cov[0][0],
            "predictive_cov_iq": predictive_cov[0][1],
            "predictive_cov_qq": predictive_cov[1][1],
            "heldout_reference_log_evidence": heldout_receipt.log_evidence,
            "heldout_proper_score_per_sample": per_complex_sample_log_score(
                heldout_receipt.log_evidence,
                sample_count=heldout_receipt.sample_count,
            ),
            "heldout_llr_ge_per_sample": heldout_receipt.pairwise_llr[0][1]
            / heldout_receipt.sample_count,
            "heldout_llr_gf_per_sample": heldout_receipt.pairwise_llr[0][2]
            / heldout_receipt.sample_count,
            "heldout_llr_ef_per_sample": heldout_receipt.pairwise_llr[1][2]
            / heldout_receipt.sample_count,
            "drift_0": result.state.drift.vector()[0],
            "drift_1": result.state.drift.vector()[1],
            "drift_2": result.state.drift.vector()[2],
            "drift_3": result.state.drift.vector()[3],
            "drift_4": result.state.drift.vector()[4],
            "density_trace_error": diagnostics[0],
            "density_hermiticity_frobenius": diagnostics[1],
            "density_minimum_eigenvalue": diagnostics[2],
            "density_quantization_frobenius_error": (
                density_quantization_frobenius_error
            ),
            "density_quantization_certified_frobenius_bound": (
                density_quantization_certified_frobenius_bound
            ),
            "density_quantization_trace_distance_bound": (
                density_quantization_trace_distance_bound
            ),
            "posterior_normalization_error": posterior_error,
            "level_normalization_error": level_error,
            "reference_posterior_l1_error": reference_posterior_error,
            "reference_log_evidence_error": reference_log_error,
            "conservation_pass": conservation,
        }
    )
    if logical is not None:
        logical_fields = (
            "logical_survival",
            "logical_block_00_real",
            "logical_block_00_imag",
            "logical_block_01_real",
            "logical_block_01_imag",
            "logical_block_10_real",
            "logical_block_10_imag",
            "logical_block_11_real",
            "logical_block_11_imag",
        )
        row.update(dict(zip(logical_fields, logical)))
    raw_iq = np.column_stack((observation.iq_i, observation.iq_q)).astype(
        np.float64, copy=False
    )
    return row, raw_iq


def _exception_row(
    identity: Mapping[str, object],
    exception: BaseException,
    heldout: np.ndarray,
) -> dict[str, object]:
    row = _empty_row(identity)
    row["heldout_window_sha256"] = _sha_bytes(
        np.asarray(heldout, dtype="<f8").tobytes(order="C")
    )
    row["exception_type"] = type(exception).__name__
    row["exception_message"] = str(exception)[:1000]
    return row


def _initial_state(
    cell: CellSpec,
    simulator: object,
) -> tuple[object, object | None]:
    if cell.layer == "logical":
        return simulator.initialize_logical(cell.logical_label)
    ket, ancilla = _initial_fock_ket(cell.initial_state or "vacuum_g", cell.cutoff)
    return simulator.initialize_fock(oscillator_ket=ket, ancilla_state=ancilla), None


def execute_cell(
    config: Mapping[str, Any],
    cell: CellSpec,
    simulator: object,
    actions: Mapping[str, ActionWord],
) -> ChunkEvidence:
    """Execute a complete denominator; exceptions become explicit rows."""

    iq_samples = int(config["common_physics"]["iq_samples"])
    rows: list[dict[str, object]] = []
    densities: list[np.ndarray] = []
    density_row_ids: list[str] = []
    raw_windows: list[np.ndarray] = []
    heldout_windows: list[np.ndarray] = []
    matrix = config["formal_matrix"]

    for position in range(cell.sample_count):
        seed = _seed_for(config, cell, position)
        if cell.layer != "fault":
            heldout = _heldout_window(
                config,
                cell_base=cell.cell_base,
                cutoff=cell.cutoff,
                position=position,
                round_index=0,
            )
            heldout_windows.append(heldout)
            identity = _identity(
                cell,
                seed=seed,
                position=position,
                row_index=len(rows),
                action=cell.action,
            )
            try:
                state, evaluator = _initial_state(cell, simulator)
                action_word = actions[cell.action]
                result = _one_step(
                    backend=cell.backend,
                    simulator=simulator,
                    state=state,
                    evaluator=evaluator,
                    action=action_word,
                    seed=seed,
                )
                retain_density = cell.layer in {"shared", "probe"}
                density_index = len(densities) if retain_density else -1
                row, raw_iq = _success_row(
                    config=config,
                    identity=identity,
                    simulator=simulator,
                    result=result,
                    action=action_word,
                    heldout=heldout,
                    density_index=density_index,
                )
                rows.append(row)
                raw_windows.append(raw_iq)
                if retain_density:
                    densities.append(
                        np.asarray(
                            result.state.joint_density, dtype=np.complex64
                        )
                    )
                    density_row_ids.append(str(row["row_id"]))
            except BaseException as exc:
                rows.append(_exception_row(identity, exc, heldout))
                raw_windows.append(np.full((iq_samples, 2), np.nan, dtype=np.float64))
            continue

        specification = matrix["fault_scenarios"][cell.scenario]
        sequence = list(matrix["fault_action_sequences"][cell.scenario])
        fault_labels = list(matrix["fault_logical_label_schedule"])
        fault_label = str(fault_labels[position % len(fault_labels)])
        try:
            state, evaluator = simulator.initialize_logical(fault_label)
            failed: BaseException | None = None
        except BaseException as exc:
            state = None
            evaluator = None
            failed = exc
        for round_index in range(cell.horizon):
            action_name = sequence[round_index % len(sequence)]
            heldout = _heldout_window(
                config,
                cell_base=cell.cell_base,
                cutoff=cell.cutoff,
                position=position,
                round_index=round_index,
            )
            heldout_windows.append(heldout)
            identity = _identity(
                cell,
                seed=seed,
                position=position,
                row_index=len(rows),
                action=action_name,
                round_index=round_index,
                terminal_round=round_index == cell.horizon - 1,
                logical_label=fault_label,
            )
            if failed is not None:
                rows.append(_exception_row(identity, failed, heldout))
                raw_windows.append(np.full((iq_samples, 2), np.nan, dtype=np.float64))
                continue
            try:
                delta = _fault_delta_for_round(
                    cell.scenario, specification, round_index
                )
                if np.any(delta):
                    state = _apply_intervention(state, cell.backend, delta)
                action_word = actions[action_name]
                result = _one_step(
                    backend=cell.backend,
                    simulator=simulator,
                    state=state,
                    evaluator=evaluator,
                    action=action_word,
                    seed=seed,
                )
                terminal = round_index == cell.horizon - 1
                density_index = len(densities) if terminal else -1
                row, raw_iq = _success_row(
                    config=config,
                    identity=identity,
                    simulator=simulator,
                    result=result,
                    action=action_word,
                    heldout=heldout,
                    density_index=density_index,
                )
                rows.append(row)
                raw_windows.append(raw_iq)
                if terminal:
                    densities.append(
                        np.asarray(result.state.joint_density, dtype=np.complex64)
                    )
                    density_row_ids.append(str(row["row_id"]))
                state = result.state
            except BaseException as exc:
                failed = exc
                rows.append(_exception_row(identity, exc, heldout))
                raw_windows.append(np.full((iq_samples, 2), np.nan, dtype=np.float64))
    if len(rows) != cell.expected_rows:
        raise RuntimeError("cell execution produced an incomplete denominator")
    return ChunkEvidence(
        rows=rows,
        densities=densities,
        density_row_ids=density_row_ids,
        raw_iq=np.stack(raw_windows),
        heldout_iq=np.stack(heldout_windows),
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_chunk(
    root: Path,
    config: Mapping[str, Any],
    cell: CellSpec,
    evidence: ChunkEvidence,
) -> dict[str, object]:
    chunk_directory = root / str(config["artifact_paths"]["chunk_directory"])
    csv_path = chunk_directory / f"{cell.chunk_id}.csv"
    npz_path = chunk_directory / f"{cell.chunk_id}.npz"
    _atomic_csv(csv_path, evidence.rows)
    dimension = cell.cutoff * 3
    densities = (
        np.stack(evidence.densities)
        if evidence.densities
        else np.empty((0, dimension, dimension), dtype=np.complex64)
    )
    row_width = max([1] + [len(value) for value in evidence.density_row_ids])
    _atomic_npz(
        npz_path,
        schema=np.asarray([RAW_ARCHIVE_SCHEMA]),
        chunk_id=np.asarray([cell.chunk_id]),
        cutoff=np.asarray([cell.cutoff], dtype=np.int64),
        row_ids=np.asarray([str(row["row_id"]) for row in evidence.rows]),
        density_row_ids=np.asarray(evidence.density_row_ids, dtype=f"<U{row_width}"),
        densities=densities,
        raw_iq=np.asarray(evidence.raw_iq, dtype=np.float64),
        heldout_iq=np.asarray(evidence.heldout_iq, dtype=np.float64),
    )
    receipt = {
        "chunk_id": cell.chunk_id,
        "cell_base": cell.cell_base,
        "layer": cell.layer,
        "backend": cell.backend,
        "cutoff": cell.cutoff,
        "expected_rows": cell.expected_rows,
        "observed_rows": len(evidence.rows),
        "exception_rows": sum(bool(row["exception_type"]) for row in evidence.rows),
        "csv": _binding(csv_path, root),
        "npz": _binding(npz_path, root),
    }
    _validate_chunk_files(root, receipt, cell)
    return receipt


def _csv_float(row: Mapping[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise RuntimeError(f"required numeric field {field} is blank")
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(f"field {field} is not numeric") from exc
    if not isfinite(result):
        raise RuntimeError(f"field {field} is non-finite")
    return result


def _validate_chunk_files(
    root: Path,
    receipt: Mapping[str, object],
    cell: CellSpec,
) -> None:
    """Independently reopen a committed chunk and verify row/array semantics."""

    csv_binding = receipt.get("csv")
    npz_binding = receipt.get("npz")
    if not isinstance(csv_binding, Mapping) or not isinstance(npz_binding, Mapping):
        raise RuntimeError("chunk receipt lacks CSV/NPZ bindings")
    csv_path = root / str(csv_binding["path"])
    npz_path = root / str(npz_binding["path"])
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEDGER_FIELDS:
            raise RuntimeError("chunk CSV field schema drift")
        rows = list(reader)
    if len(rows) != cell.expected_rows:
        raise RuntimeError("chunk CSV denominator mismatch")
    row_ids = [row["row_id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError("chunk contains duplicate row_id values")
    for index, row in enumerate(rows):
        if (
            row["row_schema"] != ROW_SCHEMA
            or row["archive_chunk"] != cell.chunk_id
            or int(row["archive_row_index"]) != index
            or row["backend"] != cell.backend
            or int(row["cutoff"]) != cell.cutoff
            or row["layer"] != cell.layer
        ):
            raise RuntimeError("chunk CSV identity/index mismatch")

    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "schema",
            "chunk_id",
            "cutoff",
            "row_ids",
            "density_row_ids",
            "densities",
            "raw_iq",
            "heldout_iq",
        }
        if set(archive.files) != required:
            raise RuntimeError("chunk NPZ dataset schema drift")
        if (
            archive["schema"].tolist() != [RAW_ARCHIVE_SCHEMA]
            or archive["chunk_id"].tolist() != [cell.chunk_id]
            or archive["cutoff"].tolist() != [cell.cutoff]
            or archive["row_ids"].tolist() != row_ids
        ):
            raise RuntimeError("chunk NPZ identity/row alignment mismatch")
        raw_iq = archive["raw_iq"]
        heldout_iq = archive["heldout_iq"]
        densities = archive["densities"]
        density_row_ids = archive["density_row_ids"].tolist()
        if (
            raw_iq.dtype != np.float64
            or heldout_iq.dtype != np.float64
            or raw_iq.shape != (cell.expected_rows, 8, 2)
            or heldout_iq.shape != raw_iq.shape
            or not np.all(np.isfinite(heldout_iq))
            or densities.dtype != np.complex64
            or densities.ndim != 3
            or densities.shape[1:] != (cell.cutoff * 3, cell.cutoff * 3)
            or not np.all(np.isfinite(densities))
        ):
            raise RuntimeError("chunk NPZ dtype/shape/finiteness mismatch")
        expected_density_ids: list[str] = []
        for index, row in enumerate(rows):
            exception = bool(row["exception_type"])
            if exception:
                if int(row["raw_iq_index"]) != -1 or not np.isnan(raw_iq[index]).all():
                    raise RuntimeError("exception row raw-IQ sentinel mismatch")
            else:
                if int(row["raw_iq_index"]) != index or not np.isfinite(raw_iq[index]).all():
                    raise RuntimeError("successful row raw-IQ alignment mismatch")
            if int(row["heldout_iq_index"]) != index:
                raise RuntimeError("heldout IQ index mismatch")
            heldout_hash = _sha_bytes(
                np.asarray(heldout_iq[index], dtype="<f8").tobytes(order="C")
            )
            if row["heldout_window_sha256"] != heldout_hash:
                raise RuntimeError("heldout IQ row hash mismatch")
            should_retain_density = (
                not exception
                and (
                    cell.layer in {"shared", "probe"}
                    or (
                        cell.layer == "fault"
                        and row["terminal_round"].lower() == "true"
                    )
                )
            )
            density_index = int(row["density_index"])
            if should_retain_density:
                if density_index != len(expected_density_ids):
                    raise RuntimeError("density index is not contiguous/aligned")
                expected_density_ids.append(row["row_id"])
                unit_roundoff = 2.0**-24
                certified = (
                    unit_roundoff
                    / (1.0 - unit_roundoff)
                    * np.linalg.norm(
                        densities[density_index].astype(np.complex128),
                        ord="fro",
                    )
                    + np.sqrt(2.0 * densities[density_index].size) * 2.0**-150
                )
                if not np.isclose(
                    _csv_float(
                        row, "density_quantization_certified_frobenius_bound"
                    ),
                    certified,
                    rtol=5.0e-13,
                    atol=0.0,
                ):
                    raise RuntimeError("density quantization certificate mismatch")
                trace_bound = 0.5 * np.sqrt(cell.cutoff * 3) * certified
                if not np.isclose(
                    _csv_float(row, "density_quantization_trace_distance_bound"),
                    trace_bound,
                    rtol=5.0e-13,
                    atol=0.0,
                ):
                    raise RuntimeError("density trace-distance bound mismatch")
                exact_diagnostic = _csv_float(
                    row, "density_quantization_frobenius_error"
                )
                if exact_diagnostic > certified:
                    raise RuntimeError("reported exact quantization error exceeds certificate")
            elif density_index != -1:
                raise RuntimeError("row retains a prohibited/unexpected density")
        if density_row_ids != expected_density_ids or len(densities) != len(
            expected_density_ids
        ):
            raise RuntimeError("density row-id archive alignment mismatch")


def _event_without_hash(event: Mapping[str, object]) -> dict[str, object]:
    value = dict(event)
    value.pop("event_sha256", None)
    return value


def _parse_attempt_ledger(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    if not path.exists():
        return [], b""
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise RuntimeError("attempt ledger has a torn final line")
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    for index, line in enumerate(payload.splitlines()):
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("attempt ledger contains invalid JSON") from exc
        if not isinstance(event, dict):
            raise RuntimeError("attempt event must be an object")
        if (
            event.get("event_schema") != ATTEMPT_SCHEMA
            or event.get("event_index") != index
            or event.get("previous_event_sha256") != previous
        ):
            raise RuntimeError("attempt ledger identity/hash-chain mismatch")
        actual = _sha_bytes(_canonical(_event_without_hash(event)))
        if event.get("event_sha256") != actual:
            raise RuntimeError("attempt event self-hash mismatch")
        previous = actual
        events.append(event)
    return events, payload


def _append_event(
    path: Path,
    event: Mapping[str, object],
) -> dict[str, object]:
    events, _ = _parse_attempt_ledger(path)
    complete = dict(event)
    complete.update(
        {
            "event_schema": ATTEMPT_SCHEMA,
            "event_index": len(events),
            "previous_event_sha256": (
                str(events[-1]["event_sha256"]) if events else "0" * 64
            ),
        }
    )
    complete["event_sha256"] = _sha_bytes(_canonical(complete))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical(complete) + b"\n")
        handle.flush()
    verified, _ = _parse_attempt_ledger(path)
    if verified[-1] != complete:
        raise RuntimeError("attempt event append verification failed")
    return complete


def _run_id(
    config_binding: Mapping[str, object],
    seal_binding: Mapping[str, object],
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "runner_id": RUNNER_ID,
                "config_sha256": config_binding["sha256"],
                "seal_sha256": seal_binding["sha256"],
            }
        )
    )


def _verify_chunk_receipt(
    root: Path,
    receipt: Mapping[str, object],
    cell: CellSpec,
) -> None:
    if (
        receipt.get("chunk_id") != cell.chunk_id
        or receipt.get("observed_rows") != cell.expected_rows
        or receipt.get("expected_rows") != cell.expected_rows
    ):
        raise RuntimeError(f"chunk receipt accounting mismatch: {cell.chunk_id}")
    for key in ("csv", "npz"):
        binding = receipt.get(key)
        if not isinstance(binding, Mapping) or "path" not in binding:
            raise RuntimeError(f"chunk receipt lacks {key} binding")
        if dict(binding) != _binding_for_relative(root, str(binding["path"])):
            raise RuntimeError(f"committed chunk drift: {cell.chunk_id}/{key}")
    _validate_chunk_files(root, receipt, cell)


def _resume_state(
    root: Path,
    config: Mapping[str, Any],
    config_binding: Mapping[str, object],
    seal_binding: Mapping[str, object],
    plan: Sequence[CellSpec],
) -> tuple[str, dict[str, dict[str, object]], list[dict[str, Any]]]:
    attempt_path = root / str(config["artifact_paths"]["attempt_ledger"])
    events, _ = _parse_attempt_ledger(attempt_path)
    run_id = _run_id(config_binding, seal_binding)
    for event in events:
        if (
            event.get("run_id") != run_id
            or event.get("config_sha256") != config_binding["sha256"]
            or event.get("seal_sha256") != seal_binding["sha256"]
        ):
            raise RuntimeError("attempt ledger belongs to another sealed run")
    committed: dict[str, dict[str, object]] = {}
    valid_ids = {cell.chunk_id: cell for cell in plan}
    for event in events:
        if event.get("event_kind") != "CHUNK_COMMITTED":
            continue
        receipt = event.get("chunk")
        if not isinstance(receipt, Mapping):
            raise RuntimeError("CHUNK_COMMITTED event lacks receipt")
        chunk_id = str(receipt.get("chunk_id"))
        if chunk_id not in valid_ids or chunk_id in committed:
            raise RuntimeError("attempt ledger has unknown/duplicate committed chunk")
        _verify_chunk_receipt(root, receipt, valid_ids[chunk_id])
        committed[chunk_id] = dict(receipt)
    return run_id, committed, events


def _write_heartbeat(
    root: Path,
    config: Mapping[str, Any],
    *,
    run_id: str,
    status: str,
    completed_cells: int,
    completed_rows: int,
    expected_cells: int,
    expected_rows: int,
    last_chunk_id: str | None,
) -> None:
    heartbeat = {
        "task_id": TASK_ID,
        "schema_version": HEARTBEAT_SCHEMA,
        "runner_id": RUNNER_ID,
        "run_id": run_id,
        "status": status,
        "completed_cells": completed_cells,
        "completed_rows": completed_rows,
        "expected_cells": expected_cells,
        "expected_rows": expected_rows,
        "last_chunk_id": last_chunk_id,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    heartbeat["heartbeat_sha256"] = _sha_bytes(_canonical(heartbeat))
    _atomic_text(
        root / str(config["artifact_paths"]["heartbeat"]),
        json.dumps(heartbeat, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _mapping_arrays(simulators: Mapping[int, Mapping[str, object]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for cutoff, pair in simulators.items():
        simulator_a = pair["A"]
        simulator_b = pair["B"]
        isometry_a = np.asarray(simulator_a._logical_engine().code_basis.isometry)
        isometry_b = np.asarray(simulator_b._comb_isometry())
        arrays.update(
            {
                f"mapping_isometry_a_cutoff_{cutoff}": isometry_a,
                f"mapping_isometry_b_cutoff_{cutoff}": isometry_b,
                f"mapping_projector_a_cutoff_{cutoff}": isometry_a @ isometry_a.conj().T,
                f"mapping_projector_b_cutoff_{cutoff}": isometry_b @ isometry_b.conj().T,
                f"mapping_captured_a_cutoff_{cutoff}": np.asarray(
                    simulator_a._logical_engine().code_basis.captured_probabilities
                ),
                f"mapping_captured_b_cutoff_{cutoff}": np.asarray(
                    simulator_b.logical_bridge_diagnostics()["captured_probabilities"]
                ),
            }
        )
    return arrays


def _write_mapping_chunk(
    root: Path,
    config: Mapping[str, Any],
    simulators: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    directory = root / str(config["artifact_paths"]["chunk_directory"])
    path = directory / "__mapping_arrays__.npz"
    _atomic_npz(
        path,
        schema=np.asarray([RAW_ARCHIVE_SCHEMA]),
        iq_reference_id=np.asarray([IQ_REFERENCE_ID]),
        **_mapping_arrays(simulators),
    )
    return _binding(path, root)


def _merge_ledger(
    root: Path,
    config: Mapping[str, Any],
    plan: Sequence[CellSpec],
    committed: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    output = root / str(config["artifact_paths"]["cell_ledger"])
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    global_row_ids: set[str] = set()
    with temporary.open("wb") as destination:
        destination.write((",".join(LEDGER_FIELDS) + "\n").encode("utf-8"))
        for cell in plan:
            receipt = committed[cell.chunk_id]
            payload = (root / str(receipt["csv"]["path"])).read_bytes()
            newline = payload.find(b"\n")
            if newline < 0:
                raise RuntimeError("chunk CSV lacks header")
            if payload[:newline].decode("utf-8") != ",".join(LEDGER_FIELDS):
                raise RuntimeError("chunk CSV schema drift")
            body = payload[newline + 1 :]
            with io.StringIO(payload.decode("utf-8"), newline="") as handle:
                chunk_rows = list(csv.DictReader(handle))
            for row in chunk_rows:
                row_id = row["row_id"]
                if row_id in global_row_ids:
                    raise RuntimeError("merged ledger has duplicate row_id")
                global_row_ids.add(row_id)
            destination.write(body)
            row_count += len(chunk_rows)
    temporary.replace(output)
    if row_count != sum(cell.expected_rows for cell in plan):
        raise RuntimeError("merged ledger row accounting mismatch")
    return _binding(output, root)


def _build_raw_archive(
    root: Path,
    config: Mapping[str, Any],
    plan: Sequence[CellSpec],
    committed: Mapping[str, Mapping[str, object]],
    mapping_binding: Mapping[str, object],
) -> dict[str, object]:
    output = root / str(config["artifact_paths"]["raw_archive"])
    temporary = output.with_suffix(output.suffix + ".tmp")
    entries = [
        {
            "chunk_id": cell.chunk_id,
            "layer": cell.layer,
            "cell_base": cell.cell_base,
            "backend": cell.backend,
            "cutoff": cell.cutoff,
            "rows": cell.expected_rows,
            "source": committed[cell.chunk_id]["npz"],
            "member": f"chunks/{cell.chunk_id}.npz",
        }
        for cell in plan
    ]
    archive_manifest = {
        "task_id": TASK_ID,
        "schema_version": RAW_ARCHIVE_SCHEMA,
        "format": "ZIP_DEFLATED members; every NPZ must load with allow_pickle=False",
        "chunk_count": len(entries),
        "row_count": sum(cell.expected_rows for cell in plan),
        "mapping_source": mapping_binding,
        "mapping_member": "mapping/mapping_arrays.npz",
        "entries": entries,
    }
    archive_manifest["analysis_sha256"] = _sha_bytes(_canonical(archive_manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr("archive_manifest.json", _canonical(archive_manifest) + b"\n")
        archive.write(
            root / str(mapping_binding["path"]),
            arcname="mapping/mapping_arrays.npz",
        )
        for entry in entries:
            archive.write(
                root / str(entry["source"]["path"]),
                arcname=str(entry["member"]),
            )
    temporary.replace(output)
    with zipfile.ZipFile(output, "r") as archive:
        if len(archive.namelist()) != len(entries) + 2:
            raise RuntimeError("final raw archive member accounting mismatch")
        live_manifest = json.loads(archive.read("archive_manifest.json"))
        if live_manifest != archive_manifest:
            raise RuntimeError("raw archive manifest mismatch")
    return _binding(output, root)


def _verify_finalized(
    root: Path,
    config: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    final_events = [event for event in events if event.get("event_kind") == "FINALIZED"]
    if not final_events:
        return None
    if len(final_events) != 1 or final_events[0] is not events[-1]:
        raise RuntimeError("attempt ledger has invalid FINALIZED placement")
    final = final_events[0]
    manifest_binding = final.get("execution_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise RuntimeError("FINALIZED lacks execution manifest binding")
    if dict(manifest_binding) != _binding_for_relative(
        root, str(manifest_binding["path"])
    ):
        raise RuntimeError("final execution manifest drift")
    manifest = _strict_json(root / str(manifest_binding["path"]))
    claims = manifest.get("claim_state")
    if (
        manifest.get("status") != FORMAL_STATUS
        or manifest.get("scientific_verdict") is not None
        or manifest.get("qualified_claim") is not None
        or not isinstance(claims, Mapping)
        or len(claims) != 15
        or any(value is not None for value in claims.values())
    ):
        raise RuntimeError("final manifest status/verdict invalid")
    for name in ("cell_ledger", "raw_archive"):
        binding = manifest.get(name)
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"final manifest lacks {name}")
        if dict(binding) != _binding_for_relative(root, str(binding["path"])):
            raise RuntimeError(f"final {name} drift")
    return manifest


def run_raw_formal(root: Path) -> dict[str, Any]:
    """Execute or exactly resume the sealed formal raw-evidence transaction."""

    root = root.resolve()
    config, config_binding = load_config(root)
    required_environment = config["runtime_environment"]
    if (
        list(sys.version_info[:3]) != required_environment["python"]
        or np.__version__ != required_environment["numpy"]
        or scipy.__version__ != required_environment["scipy"]
    ):
        raise RuntimeError("formal numerical runtime does not match sealed config")
    seal, seal_binding = verify_preformal_seal(root, config, config_binding)
    plan = build_cell_plan(config)
    run_id, committed, prior_events = _resume_state(
        root, config, config_binding, seal_binding, plan
    )
    finalized = _verify_finalized(root, config, prior_events)
    if finalized is not None:
        return finalized
    attempt_path = root / str(config["artifact_paths"]["attempt_ledger"])
    _append_event(
        attempt_path,
        {
            "event_kind": "RUN_STARTED" if not prior_events else "RESUME_STARTED",
            "task_id": TASK_ID,
            "runner_id": RUNNER_ID,
            "run_id": run_id,
            "config_sha256": config_binding["sha256"],
            "seal_sha256": seal_binding["sha256"],
            "expected_cells": len(plan),
            "expected_rows": sum(cell.expected_rows for cell in plan),
            "already_committed_cells": len(committed),
        },
    )
    simulators = {
        cutoff: build_simulators(config, cutoff)
        for cutoff in config["formal_matrix"]["cutoff_ladder"]
    }
    actions = _action_words()
    try:
        for cell in plan:
            if cell.chunk_id in committed:
                continue
            evidence = execute_cell(
                config, cell, simulators[cell.cutoff][cell.backend], actions
            )
            receipt = write_chunk(root, config, cell, evidence)
            event = _append_event(
                attempt_path,
                {
                    "event_kind": "CHUNK_COMMITTED",
                    "task_id": TASK_ID,
                    "runner_id": RUNNER_ID,
                    "run_id": run_id,
                    "config_sha256": config_binding["sha256"],
                    "seal_sha256": seal_binding["sha256"],
                    "chunk": receipt,
                },
            )
            committed[cell.chunk_id] = dict(receipt)
            completed_rows = sum(
                int(item["observed_rows"]) for item in committed.values()
            )
            _write_heartbeat(
                root,
                config,
                run_id=run_id,
                status="RUNNING",
                completed_cells=len(committed),
                completed_rows=completed_rows,
                expected_cells=len(plan),
                expected_rows=sum(item.expected_rows for item in plan),
                last_chunk_id=str(event["chunk"]["chunk_id"]),
            )
    except BaseException as exc:
        _append_event(
            attempt_path,
            {
                "event_kind": "RUN_ERROR",
                "task_id": TASK_ID,
                "runner_id": RUNNER_ID,
                "run_id": run_id,
                "config_sha256": config_binding["sha256"],
                "seal_sha256": seal_binding["sha256"],
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:2000],
            },
        )
        _write_heartbeat(
            root,
            config,
            run_id=run_id,
            status="INCOMPLETE_FAIL_CLOSED",
            completed_cells=len(committed),
            completed_rows=sum(int(item["observed_rows"]) for item in committed.values()),
            expected_cells=len(plan),
            expected_rows=sum(item.expected_rows for item in plan),
            last_chunk_id=None,
        )
        raise
    if len(committed) != len(plan):
        raise RuntimeError("formal execution ended with incomplete chunk denominator")
    mapping_binding = _write_mapping_chunk(root, config, simulators)
    ledger_binding = _merge_ledger(root, config, plan, committed)
    archive_binding = _build_raw_archive(
        root, config, plan, committed, mapping_binding
    )
    events_before_manifest, attempt_prefix = _parse_attempt_ledger(attempt_path)
    manifest = {
        "task_id": TASK_ID,
        "schema_version": MANIFEST_SCHEMA,
        "runner_id": RUNNER_ID,
        "formal": True,
        "status": FORMAL_STATUS,
        "scientific_verdict": None,
        "qualified_claim": config["claim_boundary"]["runner_qualified_claim"],
        "claim_state": config["claim_boundary"]["current_claim_state"],
        "run_id": run_id,
        "expected_cells": len(plan),
        "observed_cells": len(committed),
        "expected_rows": sum(cell.expected_rows for cell in plan),
        "observed_rows": sum(int(receipt["observed_rows"]) for receipt in committed.values()),
        "exception_rows": sum(int(receipt["exception_rows"]) for receipt in committed.values()),
        "config": config_binding,
        "preformal_seal": seal_binding,
        "preformal_seal_analysis_sha256": seal["analysis_sha256"],
        "historical_lineage_analysis_sha256": _strict_json(
            root
            / str(
                config["historical_policy"]["historical_lineage_receipt"]["path"]
            )
        )["analysis_sha256"],
        "design_power_analysis_sha256": _strict_json(
            root / str(config["design_power"]["path"])
        )["analysis_sha256"],
        "cell_ledger": ledger_binding,
        "raw_archive": archive_binding,
        "mapping_chunk": mapping_binding,
        "attempt_ledger_prefix": {
            "path": str(config["artifact_paths"]["attempt_ledger"]),
            "bytes": len(attempt_prefix),
            "sha256": _sha_bytes(attempt_prefix),
            "last_event_index": len(events_before_manifest) - 1,
            "last_event_sha256": events_before_manifest[-1]["event_sha256"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    manifest["execution_sha256"] = _sha_bytes(_canonical(manifest))
    manifest_path = root / str(config["artifact_paths"]["execution_manifest"])
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    manifest_binding = _binding(manifest_path, root)
    _append_event(
        attempt_path,
        {
            "event_kind": "FINALIZED",
            "task_id": TASK_ID,
            "runner_id": RUNNER_ID,
            "run_id": run_id,
            "config_sha256": config_binding["sha256"],
            "seal_sha256": seal_binding["sha256"],
            "execution_manifest": manifest_binding,
            "cell_ledger": ledger_binding,
            "raw_archive": archive_binding,
        },
    )
    _write_heartbeat(
        root,
        config,
        run_id=run_id,
        status=FORMAL_STATUS,
        completed_cells=len(plan),
        completed_rows=manifest["observed_rows"],
        expected_cells=len(plan),
        expected_rows=manifest["expected_rows"],
        last_chunk_id=plan[-1].chunk_id,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=_root(),
        help="repository root; no seed/count/margin/cell overrides exist",
    )
    args = parser.parse_args(argv)
    manifest = run_raw_formal(args.root)
    print(_canonical(manifest).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_SCHEMA",
    "CONFIG_PATH",
    "CONFIG_SCHEMA",
    "FORMAL_STATUS",
    "HEARTBEAT_SCHEMA",
    "LEDGER_FIELDS",
    "MANIFEST_SCHEMA",
    "PREFORMAL_SEAL_SCHEMA",
    "RAW_ARCHIVE_SCHEMA",
    "ROW_SCHEMA",
    "RUNNER_ID",
    "CellSpec",
    "ChunkEvidence",
    "build_cell_plan",
    "build_simulators",
    "execute_cell",
    "expected_cell_count",
    "expected_row_count",
    "load_config",
    "run_raw_formal",
    "validate_config",
    "verify_preformal_seal",
    "write_chunk",
]
