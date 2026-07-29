"""Outcome-blind final release gate for the T04 powered qualification.

The seal is created only after the V2 plan/seed contract, full-size resource
preflight, exact source snapshot and focused anti-simplification tests agree.
It releases one raw-evidence execution, not a scientific verdict.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    EXPECTED_CLAIM_FIELDS,
    load_config,
    plan_payload,
    runtime_source_snapshot,
)


SEAL_SCHEMA = "PHASE9-POWERED-TWIN-PREFORMAL-SEAL-V1"
VALIDATION_SCHEMA = "PHASE9-POWERED-TWIN-PREFORMAL-VALIDATION-V1"


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
        "sha256": sha256(payload).hexdigest(),
    }


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"conflicting immutable preformal artifact: {path}")
        return
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
                    f"conflicting preformal publication race: {path}"
                )
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    field: str = "analysis_sha256",
) -> None:
    claimed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if claimed != _sha(unsigned):
        raise RuntimeError(f"{field} mismatch")


def _assert_no_formal_outcome(root: Path, config: Mapping[str, Any]) -> None:
    paths = config["artifact_paths"]
    object_root = root / str(paths["object_store"])
    receipt_root = root / str(paths["receipt_directory"])
    staging_root = root / str(paths["staging_directory"])
    if object_root.exists() and (
        not object_root.is_dir()
        or any(path.is_file() for path in object_root.rglob("*"))
    ):
        raise RuntimeError("formal content object exists before preformal seal")
    if receipt_root.exists() and (
        not receipt_root.is_dir() or any(receipt_root.glob("*.json"))
    ):
        raise RuntimeError("formal receipt exists before preformal seal")
    if staging_root.exists() and (
        not staging_root.is_dir()
        or any(path.is_file() for path in staging_root.rglob("*"))
    ):
        raise RuntimeError("formal staging payload exists before preformal seal")
    for name in ("inventory", "execution_manifest", "independent_verification"):
        if (root / str(paths[name])).exists():
            raise RuntimeError(f"formal {name} exists before preformal seal")


def run_focused_validation(
    root: Path,
    config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all byte-bound T04 tests in the frozen numerical environment."""

    paths = config["artifact_paths"]
    report_path = root / str(paths["preformal_validation"])
    if report_path.exists():
        report = _strict_json(report_path)
        _verify_self_hash(report)
        if (
            report.get("schema_version") != VALIDATION_SCHEMA
            or report.get("source_snapshot_sha256")
            != snapshot["source_snapshot_sha256"]
            or report.get("returncode") != 0
            or report.get("verdict") != "PASS_FOCUSED_ANTISIMPLIFICATION"
        ):
            raise RuntimeError("existing preformal validation is not reusable")
        return report
    run_directory = (root / str(paths["run_directory"])).resolve()
    run_directory.relative_to(root.resolve())
    base_temp = run_directory / "preformal_pytest_tmp"
    if base_temp.exists():
        raise RuntimeError("preformal pytest temp already exists; archive first")
    validation_paths = [
        str(root / str(value))
        for value in config["runtime_sources"]["validation_paths"]
    ]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *validation_paths,
        "-q",
        f"--basetemp={base_temp}",
    ]
    environment = dict(os.environ)
    expected_threads = config["runtime_contract"]["preimport_thread_environment"]
    if any(environment.get(key) != value for key, value in expected_threads.items()):
        raise RuntimeError("preformal validation thread environment drifted")
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    if base_temp.exists():
        resolved = base_temp.resolve()
        resolved.relative_to(run_directory)
        shutil.rmtree(resolved)
    report: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "command": command,
        "python": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "returncode": int(completed.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "verdict": (
            "PASS_FOCUSED_ANTISIMPLIFICATION"
            if completed.returncode == 0
            else "FAIL_FOCUSED_ANTISIMPLIFICATION"
        ),
        "formal_outcomes_accessed": False,
        "claim_boundary": {
            field: None for field in EXPECTED_CLAIM_FIELDS
        },
    }
    report["analysis_sha256"] = _sha(report)
    _immutable_json(report_path, report)
    if completed.returncode != 0:
        raise RuntimeError("focused preformal validation failed")
    return report


def create_preformal_seal(root: Path) -> dict[str, Any]:
    config, config_binding = load_config(root)
    _assert_no_formal_outcome(root, config)
    plan = plan_payload(config)
    plan_sha = str(plan["canonical_plan_sha256"])
    snapshot = runtime_source_snapshot(root, config)
    paths = config["artifact_paths"]
    contract_path = root / str(paths["contract_preflight"])
    contract = _strict_json(contract_path)
    _verify_self_hash(contract)
    if (
        contract.get("schema_version")
        != "PHASE9-POWERED-TWIN-CONTRACT-PREFLIGHT-V2"
        or contract.get("status") != "PASS_OUTCOME_FREE_CONTRACT_PREFLIGHT"
        or contract.get("plan_summary", {}).get("plan_sha256") != plan_sha
        or contract.get("scientific_execution_released") is not False
    ):
        raise RuntimeError("V2 outcome-free contract preflight is not valid")
    resource_path = root / str(paths["resource_preflight"])
    resource = _strict_json(resource_path)
    _verify_self_hash(resource)
    if (
        resource.get("verdict") != "PASS_RESOURCE_PREFLIGHT"
        or resource.get("source_snapshot_sha256")
        != snapshot["source_snapshot_sha256"]
        or resource.get("config_sha256") != config_binding["sha256"]
        or resource.get("plan_sha256") != plan_sha
        or resource.get("full_size_receipt_count") != 5
        or resource.get("scientific_verdict") is not None
        or resource.get("qualified_claim") is not None
        or set(resource.get("claim_boundary", {}).values()) != {None}
    ):
        raise RuntimeError("full-size resource preflight is not valid")
    validation = run_focused_validation(root, config, snapshot)
    claims = {field: None for field in EXPECTED_CLAIM_FIELDS}
    gates = {
        "P01_v2_contract_pass": True,
        "P02_exact_518_cell_plan": plan["cell_count"] == 518,
        "P03_exact_2085888_rows": plan["row_count"] == 2_085_888,
        "P04_exact_482304_densities": (
            plan["primary_density_count"] == 482_304
        ),
        "P05_full_size_resource_pass": True,
        "P06_four_worker_peak_exercised": (
            resource.get("maximum_observed_worker_overlap") == 4
        ),
        "P07_joint_maxt_3037x199_exercised": (
            resource.get("joint_maxt_profile", {}).get("gate_count") == 3037
            and resource.get("joint_maxt_profile", {}).get("replicates") == 199
        ),
        "P08_inventory_finalize_no_copy": (
            resource.get("inventory", {}).get("monolithic_archive") is None
            and resource.get("inventory", {}).get("merged_full_csv") is None
        ),
        "P09_continuous_resource_sampling": (
            int(resource.get("resource_sample_count", 0)) >= 2
        ),
        "P10_source_snapshot_complete": (
            snapshot["runtime_source_count"] == 22
            and snapshot["validation_source_count"] == 9
        ),
        "P11_focused_tests_pass": (
            validation.get("verdict")
            == "PASS_FOCUSED_ANTISIMPLIFICATION"
        ),
        "P12_no_formal_outcome_exists": True,
        "P13_claims_remain_null": True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"preformal release gate failed: {failed}")
    seal: dict[str, Any] = {
        "schema_version": SEAL_SCHEMA,
        "task_id": str(config["task_id"]),
        "verdict": "PASS_PREFORMAL_RELEASE",
        "raw_execution_released": True,
        "scientific_verdict_released": False,
        "formal_outcomes_accessed": False,
        "config_sha256": str(config_binding["sha256"]),
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "source_snapshot": snapshot,
        "bindings": {
            "contract_preflight": _binding(contract_path, root),
            "resource_preflight": _binding(resource_path, root),
            "preformal_validation": _binding(
                root / str(paths["preformal_validation"]),
                root,
            ),
        },
        "gates": gates,
        "claim_boundary": claims,
        "scientific_verdict": None,
        "qualified_claim": None,
        "official_puviani_surpass": None,
    }
    seal["analysis_sha256"] = _sha(seal)
    _immutable_json(root / str(paths["preformal_seal"]), seal)
    return seal


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the outcome-blind T04 preformal release seal."
    )
    parser.parse_args(list(argv) if argv is not None else None)
    seal = create_preformal_seal(_root())
    print(json.dumps(seal, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
