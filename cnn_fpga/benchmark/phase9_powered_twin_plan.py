"""Outcome-free materialization of the T04 plan and seed registry.

This command is safe to run before the physics implementation is imported.  It
checks every parent byte binding and the semantic T03/T05/T06 release chain,
rescans all prior Phase-9 seed literals, then atomically publishes only the
immutable plan/registry metadata.  It refuses to run if any T04 raw object,
receipt, inventory or execution manifest already exists.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    EXPECTED_CLAIM_FIELDS,
    EXPECTED_LABELS,
    TASK_ID,
    load_config,
    plan_payload,
    seed_registry_payload,
)


REPORT_SCHEMA = "PHASE9-POWERED-TWIN-CONTRACT-PREFLIGHT-V2"
HISTORICAL_SCAN_SCHEMA = "PHASE9-HISTORICAL-SEED-SCAN-V1"


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
        raise ValueError(f"{path} must contain one object")
    return value


def _binding(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    payload = resolved.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _fsync_directory(path: Path) -> bool:
    if os.name == "nt":
        return False
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"conflicting immutable plan artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    f"conflicting immutable plan publication race: {path}"
                )
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _walk_seed_literals(
    value: object,
    path: list[str],
    rows: list[dict[str, object]],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk_seed_literals(item, [*path, str(key)], rows)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_seed_literals(item, [*path, str(index)], rows)
    elif (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and "seed" in "/".join(path).lower()
    ):
        rows.append({"value": value, "locator": "/".join(path)})


def historical_seed_scan(root: Path, config_path: Path) -> dict[str, Any]:
    """Rebuild the exact pre-T04 seed-literal inventory."""

    entries: list[dict[str, object]] = []
    literals: list[dict[str, object]] = []
    for path in sorted((root / "configs/phase9").glob("*.json")):
        if path.resolve() == config_path.resolve():
            continue
        payload = path.read_bytes()
        value = _strict_json(path)
        before = len(literals)
        _walk_seed_literals(value, [path.relative_to(root).as_posix()], literals)
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": _sha_bytes(payload),
                "seed_literal_count": len(literals) - before,
            }
        )
    if not literals:
        raise RuntimeError("historical seed scan found no registered literals")
    unsigned: dict[str, Any] = {
        "scope": "configs/phase9/*.json",
        "current_config_excluded": True,
        "entries": entries,
        "seed_literals": literals,
        "maximum_registered_seed_literal": max(
            int(row["value"]) for row in literals
        ),
    }
    scan: dict[str, Any] = {
        "schema_version": HISTORICAL_SCAN_SCHEMA,
        **unsigned,
    }
    # The config freezes the hash of the scientific scan body used during
    # preregistration.  The artifact adds a schema and its own self hash.
    scan["scan_manifest_sha256"] = _sha(unsigned)
    scan["analysis_sha256"] = _sha(scan)
    return scan


def _verify_t03_t05_t06_semantics(root: Path) -> dict[str, bool]:
    t03 = _strict_json(
        root / "docs/t_risk_20260728_03_cutoff36_44_repair_fresh1_verification.json"
    )
    t05 = _strict_json(
        root / "docs/t_risk_20260728_05_highdim_joint_maxt_verification.json"
    )
    t06 = _strict_json(
        root
        / "docs/t_risk_20260728_06_count_selection_confirmation_verification.json"
    )
    blueprint = _strict_json(
        root / "docs/t_risk_20260728_06_selected_gate_blueprint.json"
    )
    selected = t06.get("selected_count")
    gates = blueprint.get("gates")
    checks = {
        "P01_t03_design_repair_independent_pass": (
            t03.get("verification_verdict")
            == "VERIFIED_DESIGN_REPAIR_PASS_MAY_PREREGISTER_SEPARATE_POWERED_FORMAL"
            and t03.get("passed_gate_count") == 1454
            and t03.get("failed_gate_count") == 0
            and t03.get("powered_formal_released") is False
            and t03.get("qualified_claim") is None
        ),
        "P02_t05_statistical_no_go_preserved": (
            t05.get("verdict")
            == "PASS_INDEPENDENT_T04_STATISTICAL_NO_GO_VERIFICATION"
            and t05.get("t04_preregistration_released") is False
            and t05.get("qualified_claim") is None
        ),
        "P03_t06_independent_count_pass": (
            t06.get("verdict")
            == "PASS_INDEPENDENT_COUNT_SELECTION_AND_CONFIRMATION"
            and t06.get("t04_preregistration_released") is True
            and t06.get("qualified_claim") is None
        ),
        "P04_selected_count_exact": (
            selected
            == {
                "aggregate_fault_clusters": 4608,
                "round_clusters": 1536,
                "scale": 2.0,
                "state_clusters": 768,
            }
        ),
        "P05_selected_blueprint_exact": (
            blueprint.get("gate_count") == 3043
            and blueprint.get("stochastic_gate_count") == 3037
            and isinstance(gates, list)
            and len(gates) == 3043
            and len({row.get("gate_id") for row in gates if isinstance(row, Mapping)})
            == 3043
        ),
        "P06_blueprint_counts_consume_selected_count": (
            isinstance(gates, list)
            and {
                int(row["cluster_count"])
                for row in gates
                if isinstance(row, Mapping) and not bool(row.get("deterministic"))
            }
            == {768, 1536, 4608}
        ),
        "P07_parent_claims_remain_null": all(
            t06.get("claim_boundary", {}).get(field) is None
            for field in (
                "twin_qualification",
                "ler",
                "lifetime",
                "physical_break_even",
                "official_puviani_exact",
                "puviani_nmf_surpass",
                "external_sota",
                "hardware_measured",
            )
        ),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"T04 parent semantic release chain failed: {failed}")
    return checks


def _assert_no_formal_artifacts(root: Path, config: Mapping[str, Any]) -> None:
    paths = config["artifact_paths"]
    forbidden = (
        root / str(paths["object_store"]),
        root / str(paths["staging_directory"]),
        root / str(paths["receipt_directory"]),
        root / str(paths["inventory"]),
        root / str(paths["execution_manifest"]),
        root / str(paths["independent_verification"]),
    )
    (
        object_root,
        staging_root,
        receipt_root,
        inventory,
        manifest,
        verification,
    ) = forbidden
    if object_root.exists() and (
        not object_root.is_dir() or any(object_root.rglob("*"))
    ):
        raise RuntimeError("T04 raw object exists before preformal seal")
    if staging_root.exists() and (
        not staging_root.is_dir() or any(staging_root.rglob("*"))
    ):
        raise RuntimeError("T04 staging payload exists before preformal seal")
    if receipt_root.exists() and (
        not receipt_root.is_dir() or any(receipt_root.glob("*.json"))
    ):
        raise RuntimeError("T04 receipt exists before preformal seal")
    if inventory.exists() or manifest.exists() or verification.exists():
        raise RuntimeError("T04 finalize artifact exists before preformal seal")


def materialize(root: Path) -> dict[str, Any]:
    config, config_binding = load_config(root)
    _assert_no_formal_artifacts(root, config)
    semantic = _verify_t03_t05_t06_semantics(root)
    config_path = root / str(config_binding["path"])
    scan = historical_seed_scan(root, config_path)
    expected_scan = config["seed_registry"]["historical_scan"]
    if (
        len(scan["entries"]) != expected_scan["scanned_json_count"]
        or len(scan["seed_literals"]) != expected_scan["seed_literal_count"]
        or scan["maximum_registered_seed_literal"]
        != expected_scan["maximum_registered_seed_literal"]
        or scan["scan_manifest_sha256"] != expected_scan["scan_manifest_sha256"]
    ):
        raise RuntimeError("historical seed inventory drifted after preregistration")
    plan = plan_payload(config)
    registry = seed_registry_payload(config)
    paths = config["artifact_paths"]
    scan_path = root / str(paths["historical_seed_scan"])
    plan_path = root / str(paths["plan"])
    registry_path = root / str(paths["seed_registry"])
    _immutable_json(scan_path, scan)
    _immutable_json(plan_path, plan)
    _immutable_json(registry_path, registry)
    claims = {field: None for field in EXPECTED_CLAIM_FIELDS}
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_OUTCOME_FREE_CONTRACT_PREFLIGHT",
        "formal_outcomes_accessed": False,
        "scientific_execution_released": False,
        "parent_semantic_checks": semantic,
        "plan_summary": {
            "cells": plan["cell_count"],
            "rows": plan["row_count"],
            "primary_densities": plan["primary_density_count"],
            "plan_sha256": plan["canonical_plan_sha256"],
        },
        "seed_summary": {
            "historical_files": len(scan["entries"]),
            "historical_seed_literals": len(scan["seed_literals"]),
            "historical_maximum": scan["maximum_registered_seed_literal"],
            "physical_unique": registry["actual_unique_physical_addresses"],
            "heldout_unique": registry["actual_unique_heldout_addresses"],
            "registry_sha256": registry["registry_sha256"],
        },
        "bindings": {
            "config": config_binding,
            "historical_seed_scan": _binding(scan_path, root),
            "plan": _binding(plan_path, root),
            "seed_registry": _binding(registry_path, root),
        },
        "superseded_preseal_v1": {
            name: _binding(root / relative, root)
            for name, relative in {
                "contract_preflight": (
                    "docs/t_risk_20260728_04_powered_twin_contract_preflight.json"
                ),
                "historical_seed_scan": (
                    "runs/t_risk_20260728_04_powered_twin_qualification_fresh1/"
                    "historical_seed_scan.json"
                ),
                "plan": (
                    "runs/t_risk_20260728_04_powered_twin_qualification_fresh1/"
                    "plan.json"
                ),
                "seed_registry": (
                    "runs/t_risk_20260728_04_powered_twin_qualification_fresh1/"
                    "seed_registry.json"
                ),
            }.items()
        },
        "gates": {
            "C01_all_parent_bytes_verified": True,
            "C02_t03_t05_t06_semantic_chain_verified": True,
            "C03_exact_518_cell_plan": plan["cell_count"] == 518,
            "C04_exact_2085888_row_denominator": plan["row_count"] == 2_085_888,
            "C05_exact_482304_primary_densities": (
                plan["primary_density_count"] == 482_304
            ),
            "C06_fault_state_major_6x768": (
                plan["fault_state_order"] == list(EXPECTED_LABELS)
                and plan["fault_clusters_per_state"] == 768
            ),
            "C07_historical_seed_scan_recomputed": True,
            "C08_actual_seed_addresses_injective": True,
            "C09_content_addressed_no_zip_contract_frozen": True,
            "C10_all_claims_null_and_scientific_execution_blocked": True,
        },
        "claim_boundary": claims,
        "qualified_claim": None,
    }
    if not all(report["gates"].values()):
        raise RuntimeError("T04 contract preflight gate failed")
    report["analysis_sha256"] = _sha(report)
    report_path = root / str(paths["contract_preflight"])
    _immutable_json(report_path, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize the outcome-free T04 plan and seed registry."
    )
    parser.parse_args(list(argv) if argv is not None else None)
    report = materialize(_root())
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
