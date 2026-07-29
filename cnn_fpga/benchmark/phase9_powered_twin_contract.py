"""Immutable pre-outcome contract for the powered Phase-9 twin qualification.

This module is intentionally physics-free.  It owns only the pieces that must
be frozen before any T04 scientific row can exist:

* the exact 518-cell / 2,085,888-row execution plan;
* the state-major six-state fault denominator;
* injective physical, heldout and multiplier seed namespaces;
* byte-level parent evidence bindings and literal-null claim boundaries.

The scientific runner and the independent verifier consume this contract, but
neither is allowed to mutate it.  In particular, this module never imports a
physics backend, an outcome evaluator, or a prior formal archive reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping


TASK_ID = "T-RISK-20260728-04"
CONFIG_SCHEMA = "PHASE9-POWERED-TWIN-QUALIFICATION-CONFIG-V1"
PLAN_SCHEMA = "PHASE9-POWERED-TWIN-PLAN-V1"
SEED_REGISTRY_SCHEMA = "PHASE9-POWERED-TWIN-SEED-REGISTRY-V1"
CONFIG_PATH = (
    "configs/phase9/t_risk_20260728_04_powered_twin_qualification.json"
)

EXPECTED_CUTOFFS = (36, 40, 44)
EXPECTED_STATES = ("vacuum_g", "one_g", "zero_one_plus_g", "vacuum_e", "vacuum_f")
EXPECTED_LABELS = ("0", "1", "+", "-", "+i", "-i")
EXPECTED_ACTIONS = ("IDLE", "X", "Z", "XZ", "RESET", "HOLD", "LKG_HOLD")
EXPECTED_SCENARIOS = ("step", "telegraph", "burst", "compound")
EXPECTED_PROBE_ACTIONS = (
    ("P01_IDLE", "IDLE"),
    ("P02_Q_POS", "X"),
    ("P03_Q_NEG", "X"),
    ("P04_P_POS", "Z"),
    ("P05_P_NEG", "Z"),
    ("P06_ALTERNATE", "X"),
    ("P07_BOUNDARY", "XZ"),
    ("P08_PHASE", "X"),
    ("P09_LEAK_RESET", "RESET"),
    ("P10_RESET_OK", "HOLD"),
    ("P11_RESET_FAIL", "RESET"),
    ("P12_BAD_CRC", "LKG_HOLD"),
    ("P13_STALE", "LKG_HOLD"),
    ("P14_OOD", "LKG_HOLD"),
    ("P15_DEADLINE", "LKG_HOLD"),
    ("P16_LKG_RECOVERY", "LKG_HOLD"),
)
EXPECTED_FAULT_SEQUENCES = {
    "step": ("IDLE", "X", "Z", "XZ"),
    "telegraph": ("X", "Z", "IDLE"),
    "burst": ("XZ", "IDLE", "HOLD"),
    "compound": ("X", "Z", "IDLE", "XZ", "RESET", "HOLD"),
}
EXPECTED_CLAIM_FIELDS = (
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


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


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


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} escapes the repository")
    return value


def _binding(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"binding escapes repository: {path}") from exc
    payload = resolved.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


@dataclass(frozen=True)
class T04CellSpec:
    """One immutable independently committed work unit."""

    plan_index: int
    chunk_id: str
    layer: str
    cutoff: int
    backend: str
    cell_base: str
    pair_group_id: str
    pair_group_index: int
    sample_count: int
    horizon: int
    expected_rows: int
    action: str = ""
    initial_state: str = ""
    logical_label: str = ""
    probe_id: str = ""
    scenario: str = ""
    density_retention: str = "none"
    reset_estimand_scope: str = "none"

    @property
    def convergence_role(self) -> str:
        return "fresh_powered_cutoff36_40_44_qualification"


def _chunk_id(identity: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in identity)
    return f"{safe}__{sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _matrix(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("formal_matrix")
    if not isinstance(value, Mapping):
        raise ValueError("formal_matrix is missing")
    return value


def _pair_groups(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    matrix = _matrix(config)
    groups: list[tuple[str, str]] = []
    for state in matrix["shared_fock_states"]:
        for action in matrix["actions"]:
            groups.append((f"shared/{state}/{action}", "shared"))
    for label in matrix["logical_labels"]:
        for action in matrix["actions"]:
            groups.append((f"logical/{label}/{action}", "logical"))
    for probe in matrix["probe_actions"]:
        groups.append((f"probe/{probe['probe_id']}", "probe"))
    for scenario in matrix["fault_scenarios"]:
        groups.append((f"fault/{scenario}", "fault"))
    if len(groups) != 97 or len({name for name, _ in groups}) != 97:
        raise RuntimeError("pair-group accounting drift")
    return groups


def build_cell_plan(config: Mapping[str, Any]) -> list[T04CellSpec]:
    """Materialize the exact 518-cell matrix in canonical order."""

    validate_config(config, verify_plan=False)
    matrix = _matrix(config)
    groups = _pair_groups(config)
    group_index = {name: index for index, (name, _) in enumerate(groups)}
    round_count = int(matrix["round_clusters_per_cell"])
    fault_count = int(matrix["aggregate_fault_clusters_per_cell"])
    horizon = int(matrix["fault_horizon"])
    cells: list[T04CellSpec] = []

    def add(
        *,
        layer: str,
        cutoff: int,
        backend: str,
        cell_base: str,
        pair_group_id: str,
        sample_count: int,
        horizon_value: int = 1,
        action: str = "",
        initial_state: str = "",
        logical_label: str = "",
        probe_id: str = "",
        scenario: str = "",
        density_retention: str,
        reset_estimand_scope: str,
    ) -> None:
        identity = f"{layer}|c{cutoff}|{cell_base}|{backend}"
        cells.append(
            T04CellSpec(
                plan_index=len(cells),
                chunk_id=_chunk_id(identity),
                layer=layer,
                cutoff=int(cutoff),
                backend=backend,
                cell_base=cell_base,
                pair_group_id=pair_group_id,
                pair_group_index=group_index[pair_group_id],
                sample_count=sample_count,
                horizon=horizon_value,
                expected_rows=sample_count * horizon_value,
                action=action,
                initial_state=initial_state,
                logical_label=logical_label,
                probe_id=probe_id,
                scenario=scenario,
                density_retention=density_retention,
                reset_estimand_scope=reset_estimand_scope,
            )
        )

    for cutoff in matrix["cutoffs"]:
        for state in matrix["shared_fock_states"]:
            for action in matrix["actions"]:
                pair = f"shared/{state}/{action}"
                for backend in ("A", "B"):
                    add(
                        layer="shared",
                        cutoff=cutoff,
                        backend=backend,
                        cell_base=f"shared|{state}|{action}",
                        pair_group_id=pair,
                        sample_count=round_count,
                        action=action,
                        initial_state=state,
                        density_retention="all_rows",
                        reset_estimand_scope=(
                            "expected_primary_plus_conditional_sidecar"
                            if action == "RESET"
                            else "none"
                        ),
                    )
        for label in matrix["logical_labels"]:
            for action in matrix["actions"]:
                pair = f"logical/{label}/{action}"
                for backend in ("A", "B"):
                    add(
                        layer="logical",
                        cutoff=cutoff,
                        backend=backend,
                        cell_base=f"logical|{label}|{action}",
                        pair_group_id=pair,
                        sample_count=round_count,
                        action=action,
                        logical_label=label,
                        density_retention="none",
                        reset_estimand_scope=(
                            "expected_primary_plus_conditional_sidecar"
                            if action == "RESET"
                            else "none"
                        ),
                    )
        for scenario in matrix["fault_scenarios"]:
            pair = f"fault/{scenario}"
            for backend in ("A", "B"):
                add(
                    layer="fault",
                    cutoff=cutoff,
                    backend=backend,
                    cell_base=f"fault|{scenario}",
                    pair_group_id=pair,
                    sample_count=fault_count,
                    horizon_value=horizon,
                    scenario=scenario,
                    density_retention="terminal_rows",
                    reset_estimand_scope=(
                        "trajectory_expected_continuation_plus_conditional_sidecar"
                        if "RESET" in matrix["fault_action_sequences"][scenario]
                        else "none"
                    ),
                )

    probe_cutoff = int(matrix["probe_cutoff"])
    for probe in matrix["probe_actions"]:
        probe_id = str(probe["probe_id"])
        action = str(probe["action"])
        pair = f"probe/{probe_id}"
        for backend in ("A", "B"):
            add(
                layer="probe",
                cutoff=probe_cutoff,
                backend=backend,
                cell_base=f"probe|{probe_id}|{action}",
                pair_group_id=pair,
                sample_count=round_count,
                action=action,
                initial_state="vacuum_g",
                probe_id=probe_id,
                density_retention="all_rows",
                reset_estimand_scope=(
                    "expected_primary_plus_conditional_sidecar"
                    if action == "RESET"
                    else "none"
                ),
            )

    identifiers = [cell.chunk_id for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("cell plan contains duplicate chunk IDs")
    layer_counts = {
        layer: sum(cell.layer == layer for cell in cells)
        for layer in ("shared", "logical", "probe", "fault")
    }
    expected_layer_counts = {"shared": 210, "logical": 252, "probe": 32, "fault": 24}
    if layer_counts != expected_layer_counts:
        raise RuntimeError(f"cell layer accounting drift: {layer_counts}")
    if len(cells) != 518 or sum(cell.expected_rows for cell in cells) != 2_085_888:
        raise RuntimeError("T04 cell/row accounting drift")
    primary_densities = sum(
        cell.sample_count
        for cell in cells
        if cell.density_retention == "all_rows"
    ) + sum(
        cell.sample_count
        for cell in cells
        if cell.density_retention == "terminal_rows"
    )
    if primary_densities != 482_304:
        raise RuntimeError("T04 primary-density accounting drift")
    return cells


def plan_payload(config: Mapping[str, Any]) -> dict[str, object]:
    cells = build_cell_plan(config)
    payload: dict[str, object] = {
        "task_id": TASK_ID,
        "schema_version": PLAN_SCHEMA,
        "cell_count": len(cells),
        "row_count": sum(cell.expected_rows for cell in cells),
        "primary_density_count": 482_304,
        "fault_state_order": list(EXPECTED_LABELS),
        "fault_clusters_per_state": 768,
        "cells": [asdict(cell) for cell in cells],
    }
    payload["canonical_plan_sha256"] = _sha(payload["cells"])
    return payload


def _fault_position(config: Mapping[str, Any], position: int) -> tuple[str, int]:
    matrix = _matrix(config)
    per_state = int(matrix["fault_clusters_per_state"])
    total = int(matrix["aggregate_fault_clusters_per_cell"])
    if not 0 <= position < total:
        raise ValueError("fault cluster position outside frozen denominator")
    state_index, within_state = divmod(position, per_state)
    labels = tuple(matrix["logical_labels"])
    return str(labels[state_index]), within_state


def cluster_root_id(
    config: Mapping[str, Any],
    cell: T04CellSpec,
    position: int,
) -> str:
    """Return the statistical cluster identity shared only by intended pairs."""

    if not 0 <= position < cell.sample_count:
        raise ValueError("cluster position outside cell denominator")
    if cell.layer == "fault":
        label, within_state = _fault_position(config, position)
        return f"{cell.pair_group_id}/state={label}/cluster={within_state:04d}"
    return f"{cell.pair_group_id}/cluster={position:04d}"


def physical_seed(
    config: Mapping[str, Any],
    cell: T04CellSpec,
    position: int,
) -> int:
    """Injective in (backend, pair-group, position), paired across cutoffs."""

    if not 0 <= position < cell.sample_count:
        raise ValueError("cluster position outside cell denominator")
    registry = config["seed_registry"]
    maximum = int(registry["maximum_cluster_positions"])
    backend_index = 0 if cell.backend == "A" else 1
    return (
        int(registry["physical"]["start"])
        + backend_index * 97 * maximum
        + cell.pair_group_index * maximum
        + position
    )


def heldout_seed(
    config: Mapping[str, Any],
    cell: T04CellSpec,
    position: int,
    round_index: int,
) -> int:
    """Injective common-IQ seed, deliberately paired across A/B and cutoffs."""

    if not 0 <= position < cell.sample_count:
        raise ValueError("cluster position outside cell denominator")
    if not 0 <= round_index < cell.horizon:
        raise ValueError("round index outside cell horizon")
    registry = config["seed_registry"]
    maximum = int(registry["maximum_cluster_positions"])
    max_horizon = int(registry["maximum_horizon"])
    return (
        int(registry["heldout"]["start"])
        + cell.pair_group_index * maximum * max_horizon
        + position * max_horizon
        + round_index
    )


def seed_registry_payload(config: Mapping[str, Any]) -> dict[str, object]:
    """Build the small immutable address ledger (not an outcome artifact)."""

    cells = build_cell_plan(config)
    physical_values: set[int] = set()
    heldout_values: set[int] = set()
    # Enumerating all actually used addresses is intentional: arithmetic range
    # checks alone have previously missed collisions in this project.
    for cell in cells:
        for position in range(cell.sample_count):
            physical_values.add(physical_seed(config, cell, position))
            for round_index in range(cell.horizon):
                heldout_values.add(
                    heldout_seed(config, cell, position, round_index)
                )
    expected_physical = 2 * ((35 + 42 + 16) * 1536 + 4 * 4608)
    expected_heldout = (
        (35 + 42 + 16) * 1536 + 4 * 4608 * 12
    )
    if len(physical_values) != expected_physical:
        raise RuntimeError("physical seed registry is not injective")
    if len(heldout_values) != expected_heldout:
        raise RuntimeError("heldout seed registry is not injective")
    if physical_values & heldout_values:
        raise RuntimeError("physical and heldout seed namespaces overlap")
    registry = config["seed_registry"]
    payload: dict[str, object] = {
        "task_id": TASK_ID,
        "schema_version": SEED_REGISTRY_SCHEMA,
        "address_contract": {
            "physical": (
                "start + backend_index*(97*4608) + "
                "pair_group_index*4608 + cluster_position"
            ),
            "heldout": (
                "start + pair_group_index*(4608*12) + "
                "cluster_position*12 + round_index"
            ),
            "cutoff_omitted_for_registered_crn": True,
            "heldout_backend_omitted_for_registered_pairing": True,
            "fault_state_major_no_modulo": True,
        },
        "historical_scan": registry["historical_scan"],
        "namespace_intervals": {
            name: dict(registry[name])
            for name in (
                "physical",
                "heldout",
                "capability_preflight",
                "resource_preflight",
                "joint_maxt_rademacher",
            )
        },
        "actual_unique_physical_addresses": len(physical_values),
        "actual_unique_heldout_addresses": len(heldout_values),
        "physical_minimum": min(physical_values),
        "physical_maximum": max(physical_values),
        "heldout_minimum": min(heldout_values),
        "heldout_maximum": max(heldout_values),
    }
    payload["registry_sha256"] = _sha(payload)
    return payload


def _validate_parent_bindings(
    root: Path,
    config: Mapping[str, Any],
) -> None:
    parents = config.get("parent_evidence")
    if not isinstance(parents, Mapping) or not parents:
        raise ValueError("parent_evidence bindings are missing")
    paths: set[str] = set()
    for label, registered in parents.items():
        if not isinstance(label, str) or not isinstance(registered, Mapping):
            raise ValueError("invalid parent binding entry")
        relative = _relative_path(registered.get("path"), f"parent_evidence.{label}")
        if relative in paths:
            raise ValueError("duplicate parent evidence path")
        paths.add(relative)
        live = _binding(root / relative, root)
        if dict(registered) != live:
            raise RuntimeError(f"parent evidence binding drift: {label}")


def validate_config(
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    verify_plan: bool = True,
) -> None:
    """Reject any count, gate, seed, transaction or claim-boundary drift."""

    if config.get("task_id") != TASK_ID or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("T04 config identity mismatch")
    firewall = config.get("outcome_firewall")
    if (
        not isinstance(firewall, Mapping)
        or firewall.get("formal_outcomes_exist_at_freeze") is not False
        or firewall.get("formal_outcomes_accessed_at_freeze") is not False
        or firewall.get("scientific_execution_released") is not False
        or firewall.get("release_requires_preformal_seal") is not True
        or firewall.get("runner_emits_scientific_verdict") is not False
    ):
        raise ValueError("outcome firewall drift")
    matrix = _matrix(config)
    expected_matrix = {
        "cutoffs": list(EXPECTED_CUTOFFS),
        "shared_fock_states": list(EXPECTED_STATES),
        "logical_labels": list(EXPECTED_LABELS),
        "actions": list(EXPECTED_ACTIONS),
        "fault_scenarios": list(EXPECTED_SCENARIOS),
        "probe_cutoff": 36,
        "round_clusters_per_cell": 1536,
        "fault_clusters_per_state": 768,
        "fault_states_per_cell": 6,
        "aggregate_fault_clusters_per_cell": 4608,
        "fault_horizon": 12,
        "all_cells_required": True,
        "no_postselection": True,
    }
    for key, expected in expected_matrix.items():
        if matrix.get(key) != expected:
            raise ValueError(f"formal_matrix.{key} drifted")
    probes = matrix.get("probe_actions")
    if not isinstance(probes, list) or [
        (item.get("probe_id"), item.get("action"))
        for item in probes
        if isinstance(item, Mapping)
    ] != list(EXPECTED_PROBE_ACTIONS):
        raise ValueError("probe action map drifted")
    sequences = matrix.get("fault_action_sequences")
    if not isinstance(sequences, Mapping) or {
        key: tuple(value) for key, value in sequences.items()
    } != EXPECTED_FAULT_SEQUENCES:
        raise ValueError("fault action sequence drifted")
    if matrix.get("fault_scenario_parameters") != {
        "step": {
            "horizon": 12,
            "change_round": 4,
            "drift_delta": [0.07, -0.04, 0.1, -0.06, 0.04],
        },
        "telegraph": {
            "horizon": 12,
            "period": 3,
            "drift_delta": [0.05, 0.03, -0.08, 0.05, 0.03],
        },
        "burst": {
            "horizon": 12,
            "start_round": 4,
            "duration": 3,
            "drift_delta": [0.12, -0.1, 0.16, 0.12, 0.09],
        },
        "compound": {
            "horizon": 12,
            "change_round": 3,
            "burst_start": 6,
            "burst_duration": 3,
            "drift_delta": [0.1, 0.08, 0.18, -0.14, 0.12],
        },
    }:
        raise ValueError("fault scenario parameter freeze drifted")
    if matrix.get("fault_intervention_witness") != {
        "vector_order": [
            "drive_q",
            "drive_p",
            "readout_i",
            "readout_q",
            "leakage_detuning",
        ],
        "dimension": 5,
        "dtype": "<f8",
        "byte_order": "C",
        "digest": "sha256(raw_little_endian_float64_bytes)",
        "applied_rule": "True iff any component is nonzero",
        "pre_intervention_link": (
            "current pre_intervention_state_sha256 equals previous "
            "output_state_sha256"
        ),
        "zero_delta_link": (
            "input_state_sha256 equals pre_intervention_state_sha256"
        ),
    }:
        raise ValueError("fault intervention witness contract drifted")
    if matrix.get("fault_state_ordering") != "state_major_6x768_no_modulo":
        raise ValueError("fault state ordering must be state-major")
    if (
        matrix.get("logical_evaluator_carry")
        != "explicit_round_to_round_frame_state"
    ):
        raise ValueError("logical evaluator carry contract drifted")
    if matrix.get("mapping_anchor_plan_indices") != {
        "36": 0,
        "40": 162,
        "44": 324,
    }:
        raise ValueError("deterministic mapping anchor plan drifted")
    accounting = matrix.get("cell_accounting")
    if accounting != {
        "shared_chunks": 210,
        "logical_chunks": 252,
        "probe_chunks": 32,
        "fault_chunks": 24,
        "total_chunks": 518,
        "primary_rows": 2_085_888,
        "primary_densities": 482_304,
    }:
        raise ValueError("formal cell accounting drifted")

    statistics = config.get("statistics_contract")
    if (
        not isinstance(statistics, Mapping)
        or statistics.get("gate_count") != 3043
        or statistics.get("stochastic_gate_count") != 3037
        or statistics.get("exact_gate_count") != 6
        or statistics.get("multiplier_replicates") != 199
        or statistics.get("quantile_method") != "higher"
        or statistics.get("joint_family") != "all_3037_stochastic_gates"
        or statistics.get("influence_source") != "observed_paired_cluster_raw"
        or statistics.get("synthetic_influence_forbidden") is not True
        or statistics.get("pointwise_z_substitution_forbidden") is not True
        or statistics.get("aggregate_rescue_forbidden") is not True
        or statistics.get("cross_state_averaging_forbidden") is not True
        or statistics.get("analysis_cluster_rules", {}).get(
            "logical_single_state_and_ptm"
        )
        != "six_state_block_by_action_and_position_shared_multiplier_root"
        or statistics.get("conservative_cutoff_leg_rule")
        != (
            "same_signs_per_root_evaluate_A_and_B_separately_then_"
            "replicatewise_max_standardized_error_and_max_leg_bound"
        )
        or statistics.get("mapping_evidence_rule")
        != (
            "c36_c40_c44_anchor_receipts_contain_A_B_isometry_and_"
            "projector_objects"
        )
    ):
        raise ValueError("joint maxT statistics contract drifted")
    reset = config.get("reset_contract")
    if (
        not isinstance(reset, Mapping)
        or reset.get("primary_estimand")
        != "RAO_BLACKWELLIZED_EXPECTED_POST_RESET_DENSITY_AND_LEVELS_V1"
        or reset.get("all_reset_scopes_required") is not True
        or reset.get("trajectory_expected_continuation_required") is not True
        or reset.get("sampled_branch_primary") is not False
        or reset.get("conditional_branch_sidecar_required") is not True
    ):
        raise ValueError("Rao-Blackwell RESET contract drifted")
    transaction = config.get("transaction_contract")
    if (
        not isinstance(transaction, Mapping)
        or transaction.get("archive")
        != "content_addressed_immutable_objects_plus_inventory"
        or transaction.get("single_monolithic_zip_forbidden") is not True
        or transaction.get("merged_full_csv_forbidden") is not True
        or transaction.get("receipt_after_object_reopen_validation") is not True
        or transaction.get("exception_or_missing_terminal")
        != "INCOMPLETE_FAIL_CLOSED"
    ):
        raise ValueError("content-addressed transaction contract drifted")

    registry = config.get("seed_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("seed_registry missing")
    if (
        registry.get("pair_group_count") != 97
        or registry.get("maximum_cluster_positions") != 4608
        or registry.get("maximum_horizon") != 12
        or registry.get("fault_state_major") is not True
        or registry.get("modulo_addressing_forbidden") is not True
    ):
        raise ValueError("seed registry shape drifted")
    historical = registry.get("historical_scan")
    if (
        not isinstance(historical, Mapping)
        or historical.get("scope") != "configs/phase9/*.json"
        or historical.get("current_config_excluded") is not True
        or historical.get("scanned_json_count") != 39
        or historical.get("seed_literal_count") != 165
        or historical.get("maximum_registered_seed_literal") != 91_420_260_725
        or historical.get("allocation_floor") != 100_000_000_000
        or historical.get("scan_manifest_sha256")
        != "a6d3f5e9a129bc248b0692eb5ef49ef990b6d47723e722326657f15cc63b1cfd"
    ):
        raise ValueError("historical seed scan contract drifted")
    intervals: list[tuple[int, int, str]] = []
    for name in (
        "physical",
        "heldout",
        "capability_preflight",
        "resource_preflight",
        "joint_maxt_rademacher",
    ):
        spec = registry.get(name)
        if not isinstance(spec, Mapping):
            raise ValueError(f"seed namespace {name} missing")
        start = _strict_int(spec.get("start"), f"{name}.start")
        count = _strict_int(spec.get("count"), f"{name}.count", minimum=1)
        if start < int(historical["allocation_floor"]):
            raise ValueError(f"seed namespace {name} overlaps historical floor")
        intervals.append((start, start + count, name))
        if name in {"capability_preflight", "resource_preflight"}:
            expected_count = (
                7_000_000 if name == "capability_preflight" else 10_000_000
            )
            if (
                count != expected_count
                or spec.get("physical_offset") != 0
                or spec.get("heldout_offset") != 1_000_000
                or spec.get("formal_overlap") is not False
            ):
                raise ValueError(f"{name} preflight subnamespace drifted")
    for index, (start, stop, name) in enumerate(intervals):
        for other_start, other_stop, other_name in intervals[index + 1 :]:
            if max(start, other_start) < min(stop, other_stop):
                raise ValueError(f"seed namespace overlap: {name}/{other_name}")

    runtime = config.get("runtime_contract")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("max_workers") != 4
        or runtime.get("production_run_id")
        != "T04-POWERED-TWIN-FRESH1-PRODUCTION-V1"
        or runtime.get("scheduling_order")
        != "outcome_blind_cost_key_desc_then_plan_index"
        or runtime.get("automatic_cell_retry") is not False
        or runtime.get("partial_receipt_resume") is not False
        or runtime.get("launch_commit_must_equal_upstream") is not True
        or runtime.get("heartbeat_period_seconds") not in (30, 60)
        or runtime.get("preimport_thread_environment")
        != {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        or runtime.get("watchdog_at_chunk_boundary") is not True
    ):
        raise ValueError("runtime/thread contract drifted")
    environment = config.get("runtime_environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("python") != [3, 12, 7]
        or environment.get("numpy") != "1.26.4"
        or environment.get("scipy") != "1.13.1"
        or environment.get("psutil") != "5.9.0"
    ):
        raise ValueError("formal numerical runtime contract drifted")
    resources = config.get("resource_contract")
    for key, expected in {
        "maximum_wall_seconds": 1_209_600,
        "maximum_artifact_bytes": 171_798_691_840,
        "maximum_peak_rss_bytes": 34_359_738_368,
        "minimum_post_projection_free_bytes": 34_359_738_368,
    }.items():
        if not isinstance(resources, Mapping) or resources.get(key) != expected:
            raise ValueError(f"resource_contract.{key} drifted")
    if resources.get("fresh_full_size_preflight_required") is not True:
        raise ValueError("fresh full-size resource preflight is mandatory")
    expected_profiles = {
        "shared_c44_b_reset_1536",
        "logical_c44_b_reset_1536",
        "probe_c36_b_reset_1536",
        "fault_c44_b_compound_4608x12",
        "backend_a_representative",
        "four_worker_concurrent_peak",
        "joint_maxt_3037x199",
        "retained_density_physicality_full_482304",
        "inventory_finalize_no_copy",
    }
    if set(resources.get("required_profiles", ())) != expected_profiles:
        raise ValueError("resource preflight profile family drifted")
    profile_plan = resources.get("profile_plan")
    expected_concurrent = {
        "plan_indices": [389, 403, 507, 485],
        "sample_counts": [1536, 1536, 1536, 4608],
        "full_frozen_denominator": True,
    }
    expected_backend_a = {
        "plan_indices": [388],
        "sample_counts": [1536],
        "full_frozen_denominator": True,
    }
    if (
        not isinstance(profile_plan, Mapping)
        or profile_plan.get("four_worker_concurrent_peak")
        != expected_concurrent
        or profile_plan.get("backend_a_representative")
        != expected_backend_a
        or profile_plan.get("joint_maxt_3037x199")
        != {
            "gate_count": 3037,
            "replicates": 199,
            "largest_cluster_count": 4608,
            "largest_density_dimension": 132,
            "streaming_required": True,
        }
        or profile_plan.get(
            "retained_density_physicality_full_482304"
        )
        != {
            "full_retained_count": 482304,
            "largest_dimension": 132,
            "block_size": 8,
            "fixture_matrix_count": 256,
            "timed_repeats": 3,
            "full_coverage_required": True,
            "sampled": False,
        }
        or profile_plan.get("inventory_finalize_no_copy")
        != {
            "receipt_count": 5,
            "monolithic_archive_forbidden": True,
        }
    ):
        raise ValueError("resource preflight execution plan drifted")
    if verify_plan:
        resource_cells = build_cell_plan(config)
        expected_cell_signatures = {
            388: ("shared", "A", 44, "RESET", "", 1536, 1536),
            389: ("shared", "B", 44, "RESET", "", 1536, 1536),
            403: ("logical", "B", 44, "RESET", "", 1536, 1536),
            485: ("fault", "B", 44, "", "compound", 4608, 55_296),
            507: ("probe", "B", 36, "RESET", "", 1536, 1536),
        }
        for index, expected in expected_cell_signatures.items():
            cell = resource_cells[index]
            observed = (
                cell.layer,
                cell.backend,
                cell.cutoff,
                cell.action,
                cell.scenario,
                cell.sample_count,
                cell.expected_rows,
            )
            if observed != expected:
                raise ValueError(f"resource profile cell {index} drifted")

    paths = config.get("artifact_paths")
    required_paths = (
        "run_directory",
        "object_store",
        "staging_directory",
        "receipt_directory",
        "attempt_ledger",
        "owner_lock",
        "heartbeat",
        "plan",
        "seed_registry",
        "historical_seed_scan",
        "contract_preflight",
        "resource_preflight",
        "preformal_validation",
        "preformal_seal",
        "inventory",
        "execution_manifest",
        "independent_verification",
    )
    if not isinstance(paths, Mapping):
        raise ValueError("artifact_paths missing")
    normalized = [_relative_path(paths.get(name), f"artifact_paths.{name}") for name in required_paths]
    if len(set(normalized)) != len(normalized):
        raise ValueError("artifact paths must be unique")
    if any(path.endswith(".zip") for path in normalized):
        raise ValueError("monolithic ZIP path is forbidden")
    sources = config.get("runtime_sources")
    if (
        not isinstance(sources, Mapping)
        or sources.get("all_byte_bound_by_preformal_seal") is not True
        or not isinstance(sources.get("paths"), list)
        or len(sources["paths"]) != 22
        or not isinstance(sources.get("validation_paths"), list)
        or len(sources["validation_paths"]) != 9
    ):
        raise ValueError("runtime/validation source registry drifted")

    claims = config.get("claim_boundary")
    if (
        not isinstance(claims, Mapping)
        or set(claims) != set(EXPECTED_CLAIM_FIELDS)
        or any(claims[field] is not None for field in EXPECTED_CLAIM_FIELDS)
    ):
        raise ValueError("all T04 claim fields must remain literal null")
    blueprint = config.get("blueprint_binding")
    if (
        not isinstance(blueprint, Mapping)
        or blueprint.get("gate_count") != 3043
        or blueprint.get("stochastic_gate_count") != 3037
        or blueprint.get("exact_gate_count") != 6
    ):
        raise ValueError("selected gate blueprint binding drifted")
    effective = config.get("effective_blueprint_binding")
    if (
        not isinstance(effective, Mapping)
        or effective.get("gate_count") != 3043
        or effective.get("stochastic_gate_count") != 3037
        or effective.get("exact_gate_count") != 6
        or effective.get("changed_gate_count") != 16
        or effective.get("changed_fields")
        != ["cluster_count", "stage", "cluster_scope"]
        or effective.get("aggregate_fault_clusters") != 4608
    ):
        raise ValueError("effective T04 blueprint binding drifted")
    if root is not None:
        _validate_parent_bindings(root, config)
        live_blueprint = _binding(
            root / _relative_path(blueprint.get("path"), "blueprint_binding.path"),
            root,
        )
        for field in ("path", "bytes", "sha256"):
            if blueprint.get(field) != live_blueprint[field]:
                raise RuntimeError("selected gate blueprint byte binding drift")
        effective_path = root / _relative_path(
            effective.get("path"),
            "effective_blueprint_binding.path",
        )
        live_effective = _binding(effective_path, root)
        for field in ("path", "bytes", "sha256"):
            if effective.get(field) != live_effective[field]:
                raise RuntimeError("effective blueprint byte binding drift")
        effective_value = _strict_json(effective_path)
        claimed_effective = effective_value.get("analysis_sha256")
        unsigned_effective = dict(effective_value)
        unsigned_effective.pop("analysis_sha256", None)
        if (
            claimed_effective != _sha(unsigned_effective)
            or claimed_effective != effective.get("analysis_sha256")
            or effective_value.get("changed_gate_count") != 16
            or effective_value.get("gate_count") != 3043
            or len(effective_value.get("gates", ())) != 3043
        ):
            raise RuntimeError("effective blueprint semantic binding drift")
    if verify_plan:
        payload = plan_payload(config)
        expected = config.get("plan_contract")
        if (
            not isinstance(expected, Mapping)
            or expected.get("cell_count") != payload["cell_count"]
            or expected.get("row_count") != payload["row_count"]
            or expected.get("primary_density_count") != payload["primary_density_count"]
            or expected.get("canonical_plan_sha256")
            != payload["canonical_plan_sha256"]
        ):
            raise ValueError("materialized plan fingerprint drifted")


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, object]]:
    path = root / CONFIG_PATH
    config = _strict_json(path)
    validate_config(config, root=root)
    return config, _binding(path, root)


def parent_bindings(
    root: Path,
    relatives: Iterable[str],
) -> dict[str, dict[str, object]]:
    """Utility used only while preparing the immutable config."""

    result: dict[str, dict[str, object]] = {}
    for relative in relatives:
        normalized = _relative_path(relative, "parent path")
        result[normalized.replace("/", "__")] = _binding(root / normalized, root)
    return result


def runtime_source_snapshot(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Hash every production and validation source before any raw outcome."""

    sources = config.get("runtime_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("runtime_sources missing")
    runtime_paths = sources.get("paths")
    validation_paths = sources.get("validation_paths")
    if (
        sources.get("all_byte_bound_by_preformal_seal") is not True
        or not isinstance(runtime_paths, list)
        or not isinstance(validation_paths, list)
        or not runtime_paths
        or not validation_paths
    ):
        raise ValueError("runtime/validation source registry drifted")
    combined = [*runtime_paths, *validation_paths]
    normalized = [
        _relative_path(value, "runtime source path") for value in combined
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate runtime/validation source path")
    runtime_set = {
        _relative_path(value, "runtime source path") for value in runtime_paths
    }
    bindings: list[dict[str, object]] = []
    for relative in normalized:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"runtime source missing/nonregular: {relative}")
        binding = _binding(path, root)
        binding["role"] = (
            "runtime" if relative in runtime_set else "validation"
        )
        bindings.append(binding)
    config_path = root / CONFIG_PATH
    snapshot: dict[str, object] = {
        "schema_version": "PHASE9-POWERED-TWIN-SOURCE-SNAPSHOT-V1",
        "config": _binding(config_path, root),
        "bindings": bindings,
        "runtime_source_count": len(runtime_paths),
        "validation_source_count": len(validation_paths),
    }
    snapshot["source_snapshot_sha256"] = _sha(snapshot)
    return snapshot


__all__ = [
    "CONFIG_PATH",
    "CONFIG_SCHEMA",
    "EXPECTED_ACTIONS",
    "EXPECTED_CLAIM_FIELDS",
    "EXPECTED_CUTOFFS",
    "EXPECTED_LABELS",
    "EXPECTED_PROBE_ACTIONS",
    "EXPECTED_SCENARIOS",
    "EXPECTED_STATES",
    "PLAN_SCHEMA",
    "SEED_REGISTRY_SCHEMA",
    "T04CellSpec",
    "TASK_ID",
    "build_cell_plan",
    "cluster_root_id",
    "heldout_seed",
    "load_config",
    "parent_bindings",
    "physical_seed",
    "plan_payload",
    "runtime_source_snapshot",
    "seed_registry_payload",
    "validate_config",
]
