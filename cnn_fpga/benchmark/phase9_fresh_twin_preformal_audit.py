"""Outcome-blind pre-formal seal for the repaired Phase-9 twin qualification.

The seal is intentionally created only after the complete pre-formal
implementation has been committed and before any fresh formal output exists.
It imports neither physics implementation, the formal runner, nor the formal
verifier.  Inputs are checked as immutable bytes and source files are inspected
through AST/text only.

This transaction does not reinterpret or rewrite the historical T9.2.4 NO-GO.
Only the fresh, cell-data-blind lineage receipt and the fresh diagnostic/power
artifacts are consumed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence


TASK_ID = "T-RISK-20260726-01"
SCHEMA_VERSION = "PHASE9-FRESH-TWIN-PREFORMAL-SEAL-V1"
AUDIT_SCHEMA_VERSION = "PHASE9-FRESH-TWIN-PREFORMAL-AUDIT-V1"
STATUS = "PASS_FRESH_TWIN_PREFORMAL_AUDIT_SEALED"
PASS_VERDICT = "PASS_FRESH_TWIN_PREFORMAL_AUDIT"

PASS_FORMAL_VERDICT = "PASS_FRESH_TWIN_QUALIFIED"
NO_GO_FORMAL_VERDICT = "NO_GO_FRESH_TWIN_QUALIFICATION"
INCOMPLETE_FORMAL_VERDICT = "INCOMPLETE_FAIL_CLOSED"
QUALIFIED_CLAIM = "dual_backend_agreement_for_fresh_repaired_synthetic_task"

CLAIM_FIELDS = (
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

INPUT_PATHS = {
    "historical_lineage": (
        "docs/t_risk_20260726_01_historical_no_go_receipt.json"
    ),
    "iq_semantics_diagnostic": (
        "docs/t_risk_20260726_01_iq_semantics_diagnostic.json"
    ),
    "iq_semantics_source": (
        "docs/t_risk_20260726_01_iq_semantics_source_data.csv"
    ),
    "readout_power": "docs/t_risk_20260726_01_readout_power.json",
    "readout_power_source": (
        "docs/t_risk_20260726_01_readout_power_source_data.csv"
    ),
    "design_power": "docs/t_risk_20260726_01_design_power.json",
    "design_power_source": (
        "docs/t_risk_20260726_01_design_power_source_data.csv"
    ),
    "design_power_config": (
        "configs/phase9/t_risk_20260726_01_design_power.json"
    ),
    "qualification_config": (
        "configs/phase9/t_risk_20260726_01_fresh_twin_qualification.json"
    ),
}

SOURCE_PATHS = {
    "independent_reference": "physics/phase9_iq_likelihood_reference.py",
    "lineage_guard": "cnn_fpga/benchmark/phase9_fresh_twin_lineage.py",
    "iq_diagnostic": (
        "cnn_fpga/benchmark/phase9_iq_semantics_diagnostic.py"
    ),
    "readout_power": (
        "cnn_fpga/benchmark/phase9_fresh_twin_readout_power.py"
    ),
    "design_power": (
        "cnn_fpga/benchmark/phase9_fresh_twin_design_power.py"
    ),
    "formal_runner": (
        "cnn_fpga/benchmark/phase9_fresh_twin_qualification.py"
    ),
    "formal_verifier": (
        "cnn_fpga/benchmark/phase9_fresh_twin_verifier.py"
    ),
    "preformal_auditor": (
        "cnn_fpga/benchmark/phase9_fresh_twin_preformal_audit.py"
    ),
}

RUNTIME_DEPENDENCY_PATHS = (
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
)

TEST_PATHS = {
    "reference_tests": "tests/test_phase9_iq_likelihood_reference.py",
    "lineage_tests": "tests/test_phase9_fresh_twin_lineage.py",
    "diagnostic_tests": "tests/test_phase9_iq_semantics_diagnostic.py",
    "readout_power_tests": "tests/test_phase9_fresh_twin_readout_power.py",
    "design_power_tests": "tests/test_phase9_fresh_twin_design_power.py",
    "runner_tests": "tests/test_phase9_fresh_twin_qualification.py",
    "verifier_tests": "tests/test_phase9_fresh_twin_verifier.py",
    "preformal_tests": (
        "tests/test_phase9_fresh_twin_preformal_audit.py"
    ),
}

# All plausible formal artifacts are prohibited before this seal.  Names are
# deliberately fresh; no historical cell-level artifact is named or opened.
FORMAL_OUTPUT_PATHS = (
    "docs/t_risk_20260726_01_fresh_execution_manifest.json",
    "docs/t_risk_20260726_01_fresh_attempt_ledger.jsonl",
    "docs/t_risk_20260726_01_fresh_cell_ledger.csv",
    "docs/t_risk_20260726_01_fresh_raw_archive.zip",
    "docs/t_risk_20260726_01_fresh_runner_heartbeat.json",
    "docs/t_risk_20260726_01_fresh_qualification.json",
    "docs/t_risk_20260726_01_fresh_qualification_source_data.csv",
    "docs/t_risk_20260726_01_fresh_verification.json",
    "docs/t_risk_20260726_01_fresh_gate_ledger.csv",
    "docs/t_risk_20260726_01_fresh_release.json",
    "configs/phase9/t_risk_20260726_01_fresh_release_pin.json",
    "runs/t_risk_20260726_01_fresh",
)

PASS_INPUT_VERDICTS = {
    "historical_lineage": "PASS_HISTORICAL_NO_GO_LINEAGE_BOUND",
    "iq_semantics_diagnostic": "PASS_FRESH_IQ_SEMANTICS_DIAGNOSTIC",
    "readout_power": "PASS_FRESH_READOUT_EMPIRICAL_POWER",
    "design_power": "PASS_FRESH_TWIN_DESIGN_POWER",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


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
    return hashlib.sha256(payload).hexdigest()


def _sha(value: object) -> str:
    return _sha_bytes(_canonical(value))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _binding(root: Path, relative: str) -> dict[str, object]:
    normalized = relative.replace("\\", "/")
    path = (root / normalized).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"binding escapes repository: {relative}")
    payload = path.read_bytes()
    return {
        "path": normalized,
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _serialized(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result


def _analysis_hash_valid(document: Mapping[str, Any]) -> bool:
    claimed = document.get("analysis_sha256")
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        return False
    payload = dict(document)
    payload.pop("analysis_sha256", None)
    return _sha(payload) == claimed


def _csv_shape(path: Path) -> tuple[int, int, bool, bool, bool]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0, False, False, False
        rows = list(reader)
    rectangular = bool(header) and all(
        len(row) == len(header) for row in rows
    )
    unique = len(rows) == len({tuple(row) for row in rows})
    nonfinite_tokens = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}
    finite_tokens = not any(
        value.strip().lower() in nonfinite_tokens
        for row in rows
        for value in row
    )
    return len(rows), len(header), rectangular, unique, finite_tokens


def _git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _commit_binding(
    root: Path,
    commit_sha: str,
    protected_paths: Sequence[str],
) -> dict[str, object]:
    if not isinstance(commit_sha, str) or not _COMMIT_RE.fullmatch(commit_sha):
        raise ValueError("preformal commit must be an exact lowercase 40-hex SHA")
    actual = _git(root, "rev-parse", f"{commit_sha}^{{commit}}")
    if actual != commit_sha:
        raise ValueError("preformal commit is not an exact commit object")
    head = _git(root, "rev-parse", "HEAD")
    if head != commit_sha:
        raise ValueError("seal must be generated at the exact preformal HEAD")
    mismatches: list[str] = []
    for relative in protected_paths:
        worktree = (root / relative).read_bytes()
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit_sha}:{relative}"],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or completed.stdout != worktree:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(
            "protected preformal paths differ from commit: "
            + ",".join(mismatches)
        )
    return {
        "commit_sha": commit_sha,
        "head_sha": head,
        "protected_path_count": len(protected_paths),
        "all_protected_paths_exact": not mismatches,
    }


def _split_intervals(config: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    splits = config.get("splits")
    if not isinstance(splits, dict):
        return {}
    result: dict[str, tuple[int, int]] = {}
    for name, spec in splits.items():
        if not isinstance(spec, dict) or set(spec) != {"start", "count"}:
            return {}
        start = spec.get("start")
        count = spec.get("count")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or start < 0
            or count <= 0
        ):
            return {}
        result[str(name)] = (start, start + count)
    return result


def _intervals_disjoint(intervals: Mapping[str, tuple[int, int]]) -> bool:
    names = sorted(intervals)
    return all(
        max(intervals[left][0], intervals[right][0])
        >= min(intervals[left][1], intervals[right][1])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )


def _contains_all(text: str, tokens: Sequence[str]) -> bool:
    return all(token in text for token in tokens)


def _contains_none(text: str, tokens: Sequence[str]) -> bool:
    return not any(token in text for token in tokens)


def _fresh_lineage_scan_live(
    root: Path,
    lineage: Mapping[str, Any],
) -> bool:
    """Reconstruct only the fresh-source allowlist recorded by lineage.

    Historical governance files remain unopened here.  The check closes the
    stale-receipt gap caused by adding a fresh source after the lineage receipt
    was generated.
    """

    prefixes = (
        "cnn_fpga/benchmark/phase9_fresh_twin_",
        "cnn_fpga/benchmark/phase9_iq_semantics_diagnostic.py",
        "physics/phase9_iq_likelihood_reference.py",
        "configs/phase9/t_risk_20260726_01_",
    )
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and any(
            path.relative_to(root).as_posix().startswith(prefix)
            for prefix in prefixes
        )
    )
    scan = lineage.get("fresh_source_scan")
    return (
        isinstance(scan, dict)
        and scan.get("scanned_paths") == actual
        and scan.get("violations") == []
    )


def _formal_fixture(
    branch: str,
) -> dict[str, Any]:
    claims = {field: None for field in CLAIM_FIELDS}
    if branch == "pass":
        return {
            "verdict": PASS_FORMAL_VERDICT,
            "complete_finite_denominator": True,
            "all_main_gates_passed": True,
            "infrastructure_errors": [],
            "qualified_claim": QUALIFIED_CLAIM,
            "claim_state": claims,
            "released_tasks": list(DOWNSTREAM_TASKS),
            "blocked_tasks": [],
            "outcome_specific_audit_rule": None,
        }
    if branch == "no_go":
        return {
            "verdict": NO_GO_FORMAL_VERDICT,
            "complete_finite_denominator": True,
            "all_main_gates_passed": False,
            "infrastructure_errors": [],
            "qualified_claim": None,
            "claim_state": claims,
            "released_tasks": [],
            "blocked_tasks": list(DOWNSTREAM_TASKS),
            "outcome_specific_audit_rule": None,
        }
    if branch == "incomplete":
        return {
            "verdict": INCOMPLETE_FORMAL_VERDICT,
            "complete_finite_denominator": False,
            "all_main_gates_passed": False,
            "infrastructure_errors": ["fixture_missing_row"],
            "qualified_claim": None,
            "claim_state": claims,
            "released_tasks": [],
            "blocked_tasks": list(DOWNSTREAM_TASKS),
            "outcome_specific_audit_rule": None,
        }
    raise ValueError(f"unknown fixture branch: {branch}")


def audit_outcome_fixture(
    fixture: Mapping[str, Any],
) -> dict[str, bool]:
    """Audit a hypothetical outcome without observing the formal outcome."""

    verdict = fixture.get("verdict")
    claims = fixture.get("claim_state")
    released = fixture.get("released_tasks")
    blocked = fixture.get("blocked_tasks")
    errors = fixture.get("infrastructure_errors")
    exact_claims = (
        isinstance(claims, dict)
        and tuple(claims) == CLAIM_FIELDS
        and all(value is None for value in claims.values())
    )
    branch_known = verdict in {
        PASS_FORMAL_VERDICT,
        NO_GO_FORMAL_VERDICT,
        INCOMPLETE_FORMAL_VERDICT,
    }
    pass_branch = verdict != PASS_FORMAL_VERDICT or (
        fixture.get("complete_finite_denominator") is True
        and fixture.get("all_main_gates_passed") is True
        and errors == []
        and fixture.get("qualified_claim") == QUALIFIED_CLAIM
        and released == list(DOWNSTREAM_TASKS)
        and blocked == []
    )
    no_go_branch = verdict != NO_GO_FORMAL_VERDICT or (
        fixture.get("complete_finite_denominator") is True
        and fixture.get("all_main_gates_passed") is False
        and errors == []
        and fixture.get("qualified_claim") is None
        and released == []
        and blocked == list(DOWNSTREAM_TASKS)
    )
    incomplete_branch = verdict != INCOMPLETE_FORMAL_VERDICT or (
        fixture.get("complete_finite_denominator") is False
        and fixture.get("all_main_gates_passed") is False
        and isinstance(errors, list)
        and bool(errors)
        and fixture.get("qualified_claim") is None
        and released == []
        and blocked == list(DOWNSTREAM_TASKS)
    )
    return {
        "F01_known_three_way_verdict": branch_known,
        "F02_exact_fifteen_typed_null": exact_claims,
        "F03_pass_branch_contract": pass_branch,
        "F04_no_go_branch_contract": no_go_branch,
        "F05_incomplete_branch_contract": incomplete_branch,
        "F06_no_outcome_specific_audit_hardcode": (
            fixture.get("outcome_specific_audit_rule") is None
        ),
    }


def _all_source_bindings(root: Path) -> dict[str, dict[str, object]]:
    """Bind direct sources and every repository-local runtime dependency.

    Both the in-memory snapshot and the persisted audit must use this single
    assembler.  Otherwise an audit can count transitive dependencies while
    silently dropping them from the seal that the formal runner consumes.
    """

    bindings = {
        name: _binding(root, relative)
        for name, relative in SOURCE_PATHS.items()
    }
    for index, relative in enumerate(RUNTIME_DEPENDENCY_PATHS):
        name = f"runtime_dependency_{index:02d}"
        runtime_binding = _binding(root, relative)
        existing = bindings.get(name)
        if existing is not None and existing != runtime_binding:
            raise ValueError(f"conflicting runtime dependency binding: {name}")
        bindings[name] = runtime_binding
    return bindings


def build_snapshot(
    root: Path,
    *,
    preformal_commit_sha: str,
) -> dict[str, Any]:
    root = root.resolve()
    documents = {
        name: _load_json(root / relative)
        for name, relative in INPUT_PATHS.items()
        if relative.endswith(".json")
    }
    config = documents["design_power_config"]
    qualification_config = documents["qualification_config"]
    lineage = documents["historical_lineage"]
    diagnostic = documents["iq_semantics_diagnostic"]
    readout = documents["readout_power"]
    design = documents["design_power"]

    intervals = _split_intervals(config)
    source_bindings = _all_source_bindings(root)
    runtime_spec = qualification_config.get("runtime_dependencies", {})
    runtime_bindings = {
        name: binding
        for name, binding in source_bindings.items()
        if name.startswith("runtime_dependency_")
    }
    test_bindings = {
        name: _binding(root, relative)
        for name, relative in TEST_PATHS.items()
    }
    input_bindings = {
        name: _binding(root, relative)
        for name, relative in INPUT_PATHS.items()
    }
    protected_paths = sorted(
        {
            *(binding["path"] for binding in source_bindings.values()),
            *(binding["path"] for binding in test_bindings.values()),
            *(binding["path"] for binding in input_bindings.values()),
        }
    )
    commit = _commit_binding(root, preformal_commit_sha, protected_paths)

    runner_path = root / SOURCE_PATHS["formal_runner"]
    verifier_path = root / SOURCE_PATHS["formal_verifier"]
    reference_path = root / SOURCE_PATHS["independent_reference"]
    auditor_path = root / SOURCE_PATHS["preformal_auditor"]
    runner_text = runner_path.read_text(encoding="utf-8")
    verifier_text = verifier_path.read_text(encoding="utf-8")
    reference_text = reference_path.read_text(encoding="utf-8")
    auditor_text = auditor_path.read_text(encoding="utf-8")
    runner_imports = _imports(runner_path)
    verifier_imports = _imports(verifier_path)
    auditor_imports = _imports(auditor_path)
    reference_imports = _imports(reference_path)

    formal_existing = [
        relative
        for relative in FORMAL_OUTPUT_PATHS
        if (root / relative).exists()
    ]
    source_shapes = {
        name: _csv_shape(root / relative)
        for name, relative in INPUT_PATHS.items()
        if relative.endswith(".csv")
    }

    families = config.get("families", {})
    margins = config.get("margins", {})
    procedure = config.get("statistical_procedure", {})
    power = config.get("power_model", {})
    matrix = config.get("formal_matrix", {})
    qualification_matrix = qualification_config.get("formal_matrix", {})
    readout_convention = config.get("readout_convention", {})
    selected = design.get("selected_sample_counts", {})
    config_sha = input_bindings["design_power_config"]["sha256"]
    gate_blueprint_contract = qualification_config.get(
        "gate_blueprint", {}
    )
    verification_contract = qualification_config.get(
        "verification_contract", {}
    )
    gate_blueprint = (
        gate_blueprint_contract.get("rows", [])
        if isinstance(gate_blueprint_contract, dict)
        else []
    )
    blueprint_ids = [
        row.get("gate_id")
        for row in gate_blueprint
        if isinstance(row, dict)
    ]
    blueprint_payload_sha = _sha(gate_blueprint)
    design_blueprint_sha = design.get("blueprint", {}).get("sha256")

    fixtures = {
        branch: audit_outcome_fixture(_formal_fixture(branch))
        for branch in ("pass", "no_go", "incomplete")
    }

    old_no_go = lineage.get("historical_bindings", {}).get(
        "formal_report", {}
    )
    raw_old_names = (
        "t9_2_4_dual_backend_" + "cell_ledger.csv",
        "t9_2_4_dual_backend_" + "qualification_source_data.csv",
        "t9_2_4_dual_backend_" + "state_archive.npz",
    )
    fresh_source_text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in SOURCE_PATHS.values()
    )

    return {
        "task_id": config.get("task_id"),
        "lineage_task_id": lineage.get("task_id"),
        "lineage_verdict": lineage.get("verdict"),
        "lineage_analysis_valid": _analysis_hash_valid(lineage),
        "lineage_fresh_scan_live": _fresh_lineage_scan_live(root, lineage),
        "historical_parent_rewritten": lineage.get(
            "historical_parent_rewritten"
        ),
        "historical_formal_verdict": old_no_go.get("verdict"),
        "fresh_old_cell_reference_count": sum(
            fresh_source_text.count(name) for name in raw_old_names
        ),
        "diagnostic_verdict": diagnostic.get("verdict"),
        "diagnostic_analysis_valid": _analysis_hash_valid(diagnostic),
        "diagnostic_old_cell_accessed": diagnostic.get(
            "old_formal_cell_data_accessed"
        ),
        "diagnostic_old_outcome_selected_margin": diagnostic.get(
            "margin_or_threshold_selected_from_old_outcome"
        ),
        "readout_verdict": readout.get("verdict"),
        "readout_analysis_valid": _analysis_hash_valid(readout),
        "readout_old_cell_accessed": readout.get(
            "historical_formal_cell_data_accessed"
        ),
        "readout_formal_pool_accessed": readout.get(
            "formal_seed_pool_accessed"
        ),
        "design_verdict": design.get("verdict"),
        "design_analysis_valid": _analysis_hash_valid(design),
        "design_old_cell_accessed": design.get(
            "historical_formal_cell_data_accessed"
        ),
        "design_old_outcome_used": design.get(
            "old_outcome_used_to_choose_design"
        ),
        "design_formal_pool_accessed": design.get(
            "formal_seed_values_accessed"
        ),
        "config_hash_matches_design": (
            design.get("bindings", {}).get("config", {}).get("sha256")
            == config_sha
        ),
        "config_hash_matches_readout": (
            readout.get("config_sha256") == config_sha
        ),
        "qualification_binds_design_report": (
            qualification_config.get("design_power", {}).get("path")
            == INPUT_PATHS["design_power"]
            and qualification_config.get("design_power", {}).get(
                "schema_version"
            )
            == design.get("schema_version")
            and qualification_config.get("design_power", {}).get(
                "required_verdict"
            )
            == design.get("verdict")
            and qualification_config.get("design_power", {}).get(
                "round_sample_count"
            )
            == design.get("selected_sample_counts", {}).get(
                "round_sample_count"
            )
            and qualification_config.get("design_power", {}).get(
                "trajectory_sample_count"
            )
            == design.get("selected_sample_counts", {}).get(
                "trajectory_sample_count"
            )
            and qualification_config.get("design_power", {}).get(
                "formal_seed_pool_accessed_by_design"
            )
            is False
            and qualification_config.get("gate_blueprint", {}).get(
                "source_design_blueprint_sha256"
            )
            == design.get("blueprint", {}).get("sha256")
        ),
        "runtime_dependency_contract_exact": (
            isinstance(runtime_spec, dict)
            and runtime_spec.get(
                "all_must_be_byte_bound_by_preformal_seal"
            )
            is True
            and runtime_spec.get("paths")
            == list(RUNTIME_DEPENDENCY_PATHS)
            and isinstance(runtime_spec.get("policy"), str)
            and "every direct runner dependency" in runtime_spec["policy"]
            and "transitive physics dependency" in runtime_spec["policy"]
            and "fail-closed" in runtime_spec["policy"]
        ),
        "runtime_dependency_binding_count": len(runtime_bindings),
        "gate_blueprint_count": len(gate_blueprint),
        "gate_blueprint_ids_unique": (
            len(blueprint_ids) == len(set(blueprint_ids))
            and all(isinstance(value, str) and value for value in blueprint_ids)
        ),
        "gate_blueprint_rows_complete": all(
            isinstance(row, dict)
            and {
                "gate_id",
                "family",
                "metric",
                "margin",
                "direction",
                "stage",
            }.issubset(row)
            for row in gate_blueprint
        ),
        "gate_blueprint_directions_exact": all(
            isinstance(row, dict)
            and row.get("direction")
            == (
                "lower"
                if row.get("metric") == "principal_singular"
                else "upper"
            )
            for row in gate_blueprint
        ),
        "gate_blueprint_normalized_sd_valid": all(
            isinstance(row, dict)
            and isinstance(row.get("normalized_sd"), (int, float))
            and not isinstance(row.get("normalized_sd"), bool)
            and math.isfinite(float(row["normalized_sd"]))
            and float(row["normalized_sd"]) > 0.0
            for row in gate_blueprint
        ),
        "gate_blueprint_declared_row_count": (
            gate_blueprint_contract.get("row_count")
            if isinstance(gate_blueprint_contract, dict)
            else None
        ),
        "gate_blueprint_source_config": (
            gate_blueprint_contract.get("source_design_config")
            if isinstance(gate_blueprint_contract, dict)
            else None
        ),
        "verification_contract_exact": (
            isinstance(verification_contract, dict)
            and verification_contract.get("gate_blueprint_ref")
            == "#/gate_blueprint/rows"
            and verification_contract.get("blueprint_sha256_ref")
            == "#/gate_blueprint/canonical_blueprint_sha256"
            and verification_contract.get(
                "source_design_blueprint_sha256"
            )
            == design_blueprint_sha
            and verification_contract.get("cluster_unit")
            == "seed_position"
            and verification_contract.get("global_test")
            == "intersection_union_equivalence"
            and verification_contract.get("cell_test")
            == "two_one_sided_tests"
            and verification_contract.get("cell_confidence_interval")
            == 0.90
            and verification_contract.get("tost_z")
            == 1.6448536269514722
            and verification_contract.get("raw_log_evidence_primary")
            is False
            and verification_contract.get("fault_mixed_unit_composite")
            is False
            and verification_contract.get("drift_normalization")
            == [0.12, 0.10, 0.18, 0.14, 0.12]
            and verification_contract.get("aggregate_rescue_forbidden")
            is True
            and verification_contract.get(
                "missing_nonfinite_exception"
            )
            == INCOMPLETE_FORMAL_VERDICT
            and verification_contract.get(
                "density_quantization_bound_must_be_added"
            )
            is True
        ),
        "gate_blueprint_sha256": blueprint_payload_sha,
        "gate_blueprint_declared_sha256": (
            gate_blueprint_contract.get("canonical_blueprint_sha256")
            if isinstance(gate_blueprint_contract, dict)
            else None
        ),
        "gate_blueprint_design_sha256": design_blueprint_sha,
        "gate_blueprint_declared_design_sha256": (
            gate_blueprint_contract.get("source_design_blueprint_sha256")
            if isinstance(gate_blueprint_contract, dict)
            else None
        ),
        "all_source_csv_nonempty_rectangular": all(
            rows > 0 and columns > 1 and rectangular
            for rows, columns, rectangular, _, _ in source_shapes.values()
        ),
        "all_source_csv_rows_unique": all(
            unique for _, _, _, unique, _ in source_shapes.values()
        ),
        "all_source_csv_no_nonfinite_tokens": all(
            finite for _, _, _, _, finite in source_shapes.values()
        ),
        "split_count": len(intervals),
        "splits_disjoint": _intervals_disjoint(intervals),
        "fresh_seed_namespace": (
            bool(intervals)
            and min(start for start, _ in intervals.values()) >= 1_000_000
        ),
        "formal_round_pool": (
            intervals.get("formal_round_backend_a"),
            intervals.get("formal_round_backend_b"),
        ),
        "formal_trajectory_pool": (
            intervals.get("formal_trajectory_backend_a"),
            intervals.get("formal_trajectory_backend_b"),
        ),
        "selected_round_count": selected.get("round_sample_count"),
        "selected_trajectory_count": selected.get(
            "trajectory_sample_count"
        ),
        "candidate_round_counts": config.get(
            "candidate_sample_counts", {}
        ).get("round"),
        "candidate_trajectory_counts": config.get(
            "candidate_sample_counts", {}
        ).get("trajectory"),
        "pilot_allowed_fields": config.get("historical_policy", {}).get(
            "pilot_may_choose"
        ),
        "old_outcome_policy": config.get("historical_policy"),
        "family_count": len(families) if isinstance(families, dict) else 0,
        "family_names": sorted(families) if isinstance(families, dict) else [],
        "all_family_metrics_have_margin": (
            isinstance(families, dict)
            and isinstance(margins, dict)
            and all(
                metric in margins
                for spec in families.values()
                if isinstance(spec, dict)
                for metric in spec.get("metrics", [])
            )
        ),
        "margin_count": len(margins) if isinstance(margins, dict) else 0,
        "all_margins_positive_finite_with_source": (
            isinstance(margins, dict)
            and bool(margins)
            and all(
                isinstance(spec, dict)
                and isinstance(spec.get("value"), (int, float))
                and not isinstance(spec.get("value"), bool)
                and math.isfinite(float(spec["value"]))
                and float(spec["value"]) > 0.0
                and isinstance(spec.get("source"), str)
                and bool(spec["source"])
                for spec in margins.values()
            )
        ),
        "all_cells_required": matrix.get("all_cells_required"),
        "no_postselection": matrix.get("no_postselection"),
        "fault_logical_initialization_exact": (
            qualification_matrix.get("fault_initialization")
            == "logical_six_state_balanced_cycle"
            and qualification_matrix.get("fault_logical_label_schedule")
            == ["0", "1", "+", "-", "+i", "-i"]
            and qualification_matrix.get("fault_label_balance")
            == (
                "label = schedule[seed_position mod 6]; every label receives "
                "42 or 43 of 256 trajectories and the same position/label "
                "is used across A/B and cutoffs"
            )
        ),
        "cutoff_ladder": matrix.get("cutoff_ladder"),
        "primary_cutoff_increments": matrix.get(
            "primary_cutoff_increments"
        ),
        "global_test": procedure.get("global_test"),
        "cell_test": procedure.get("cell_test"),
        "cell_confidence_interval": procedure.get(
            "cell_confidence_interval"
        ),
        "global_type_i_error": procedure.get("global_type_i_error"),
        "mixed_unit_vector_max": procedure.get("mixed_unit_vector_max"),
        "raw_log_evidence_policy": procedure.get("raw_log_evidence"),
        "cell_deletion": procedure.get("cell_deletion"),
        "mean_only_rescue": procedure.get("mean_only_rescue"),
        "nonfinite_policy": procedure.get("missing_nonfinite_exception"),
        "cluster_unit": procedure.get("cluster_unit"),
        "power_pseudoexperiments": power.get("pseudoexperiments"),
        "power_formal_pool_access": power.get("formal_pool_access"),
        "power_false_pass_ucb": power.get(
            "outside_margin_false_pass_ucb_maximum"
        ),
        "power_same_a_lcb": power.get(
            "same_backend_global_equivalence_power_lcb_minimum"
        ),
        "power_same_b_covered": design.get("gates", {}).get(
            "G16_same_backend_b_power_covered"
        ),
        "power_ab_lcb": power.get(
            "ab_zero_effect_global_equivalence_power_lcb_minimum"
        ),
        "power_mutants_covered": design.get("gates", {}).get(
            "G18_all_alternative_mutants_covered"
        ),
        "readout_base_measure": readout_convention.get("raw_base_measure"),
        "sigma_convention": readout_convention.get("sigma"),
        "integration_convention": readout_convention.get("integration"),
        "latent_conditioning": readout_convention.get(
            "latent_conditioning"
        ),
        "gain_jacobian_rule": readout_convention.get(
            "gain_jacobian_rule"
        ),
        "proper_score_unit": readout_convention.get("proper_score_unit"),
        "reference_no_backend_import": not any(
            name.startswith("physics.phase9_backend_")
            for name in reference_imports
        ),
        "reference_no_random_or_numpy_import": (
            "random" not in reference_imports
            and "numpy" not in reference_imports
        ),
        "reference_formula_tokens": _contains_all(
            reference_text,
            (
                "log_evidence",
                "posterior",
                "residual",
                "log_likelihood_ratio",
                "predictive_moments",
                "marginal_cdf",
            ),
        ),
        "runner_imports_backend_a": any(
            name == "physics.phase9_backend_a" for name in runner_imports
        ),
        "runner_imports_backend_b": any(
            name == "physics.phase9_backend_b" for name in runner_imports
        ),
        "runner_no_cli_formal_overrides": _contains_none(
            runner_text,
            (
                'add_argument("--seed',
                'add_argument("--margin',
                'add_argument("--tolerance',
                'add_argument("--repeats',
                'add_argument("--sample-count',
            ),
        ),
        "runner_receipt_exact_tokens": _contains_all(
            runner_text,
            (
                "verify_preformal_seal",
                "live_bindings",
                "analysis_sha256",
                "sha256",
            ),
        ),
        "runner_complete_attempt_tokens": _contains_all(
            runner_text,
            (
                "attempt",
                "expected",
                "observed",
                "exception",
            ),
        ),
        "runner_fault_logical_tokens": _contains_all(
            runner_text,
            (
                'matrix["fault_logical_label_schedule"]',
                "position % len(fault_labels)",
                "simulator.initialize_logical(fault_label)",
                "logical_label=fault_label",
                "evaluator=evaluator",
            ),
        ),
        "verifier_no_physics_import": not any(
            name.startswith("physics") for name in verifier_imports
        ),
        "verifier_three_verdict_tokens": _contains_all(
            verifier_text,
            (
                PASS_FORMAL_VERDICT,
                NO_GO_FORMAL_VERDICT,
                INCOMPLETE_FORMAL_VERDICT,
            ),
        ),
        "verifier_claim_tokens": _contains_all(
            verifier_text, (*CLAIM_FIELDS, QUALIFIED_CLAIM)
        ),
        "verifier_release_tokens": _contains_all(
            verifier_text, DOWNSTREAM_TASKS
        ),
        "verifier_iut_tokens": _contains_all(
            verifier_text,
            (
                "intersection_union",
                "two_one_sided",
                "1.6448536269514722",
            ),
        ),
        "verifier_no_outcome_hardcode": _contains_none(
            verifier_text.lower(),
            (
                "expected_formal_verdict",
                "force_pass",
                "force_no_go",
                "assume_pass",
            ),
        ),
        "verifier_seal_policy_tokens": _contains_all(
            verifier_text,
            (
                "fresh_config",
                "historical_formal_cell_data_accessed",
                "all_mutations_detected",
                "scientific_verdict",
            ),
        ),
        "verifier_terminal_logical_survival_token": (
            "terminal_logical_survival" in verifier_text
        ),
        "auditor_no_runtime_import": not any(
            name.startswith("physics")
            or "phase9_fresh_twin_qualification" in name
            or "phase9_fresh_twin_verifier" in name
            for name in auditor_imports
        ),
        "auditor_outcome_blind_fixture_tokens": _contains_all(
            auditor_text,
            (
                'branch == "pass"',
                'branch == "no_go"',
                'branch == "incomplete"',
            ),
        ),
        "formal_existing": formal_existing,
        "commit_sha": commit["commit_sha"],
        "commit_head_exact": commit["head_sha"] == commit["commit_sha"],
        "commit_paths_exact": commit["all_protected_paths_exact"],
        "protected_path_count": commit["protected_path_count"],
        "fixture_pass": all(fixtures["pass"].values()),
        "fixture_no_go": all(fixtures["no_go"].values()),
        "fixture_incomplete": all(fixtures["incomplete"].values()),
    }


def audit_snapshot(snapshot: Mapping[str, Any]) -> dict[str, bool]:
    """Apply the frozen pre-formal gates to a serializable snapshot."""

    round_pools = snapshot.get("formal_round_pool")
    trajectory_pools = snapshot.get("formal_trajectory_pool")
    return {
        "G01_task_identity": (
            snapshot.get("task_id") == TASK_ID
            and snapshot.get("lineage_task_id") == TASK_ID
        ),
        "G02_historical_lineage_pass": (
            snapshot.get("lineage_verdict")
            == PASS_INPUT_VERDICTS["historical_lineage"]
            and snapshot.get("lineage_analysis_valid") is True
        ),
        "G03_historical_no_go_not_rewritten": (
            snapshot.get("historical_parent_rewritten") is False
            and snapshot.get("historical_formal_verdict")
            == "NO_GO_TWIN_QUALIFICATION"
        ),
        "G04_no_historical_cell_artifact_reference": (
            snapshot.get("fresh_old_cell_reference_count") == 0
        ),
        "G05_iq_diagnostic_pass": (
            snapshot.get("diagnostic_verdict")
            == PASS_INPUT_VERDICTS["iq_semantics_diagnostic"]
            and snapshot.get("diagnostic_analysis_valid") is True
        ),
        "G06_iq_diagnostic_outcome_blind": (
            snapshot.get("diagnostic_old_cell_accessed") is False
            and snapshot.get("diagnostic_old_outcome_selected_margin") is False
        ),
        "G07_readout_power_pass": (
            snapshot.get("readout_verdict")
            == PASS_INPUT_VERDICTS["readout_power"]
            and snapshot.get("readout_analysis_valid") is True
        ),
        "G08_readout_power_outcome_blind": (
            snapshot.get("readout_old_cell_accessed") is False
            and snapshot.get("readout_formal_pool_accessed") is False
        ),
        "G09_design_power_pass": (
            snapshot.get("design_verdict")
            == PASS_INPUT_VERDICTS["design_power"]
            and snapshot.get("design_analysis_valid") is True
        ),
        "G10_design_power_outcome_blind": (
            snapshot.get("design_old_cell_accessed") is False
            and snapshot.get("design_old_outcome_used") is False
            and snapshot.get("design_formal_pool_accessed") is False
        ),
        "G11_config_evidence_hash_chain": (
            snapshot.get("config_hash_matches_design") is True
            and snapshot.get("config_hash_matches_readout") is True
        ),
        "G12_source_tables_complete": (
            snapshot.get("all_source_csv_nonempty_rectangular") is True
        ),
        "G13_all_seed_splits_present": snapshot.get("split_count") == 16,
        "G14_seed_splits_disjoint": snapshot.get("splits_disjoint") is True,
        "G15_fresh_seed_namespace": (
            snapshot.get("fresh_seed_namespace") is True
        ),
        "G16_formal_round_pools_exact": (
            round_pools == ((1_070_000, 1_070_768), (1_071_000, 1_071_768))
        ),
        "G17_formal_trajectory_pools_exact": (
            trajectory_pools
            == ((1_072_000, 1_072_256), (1_073_000, 1_073_256))
        ),
        "G18_selected_power_counts_exact": (
            snapshot.get("selected_round_count") == 768
            and snapshot.get("selected_trajectory_count") == 256
        ),
        "G19_candidate_sets_frozen": (
            snapshot.get("candidate_round_counts") == [128, 256, 512, 768]
            and snapshot.get("candidate_trajectory_counts") == [96, 192, 256]
        ),
        "G20_pilot_selects_counts_only": snapshot.get(
            "pilot_allowed_fields"
        )
        == ["round_sample_count", "trajectory_sample_count"],
        "G21_old_outcome_cannot_choose_design": snapshot.get(
            "old_outcome_policy"
        )
        == {
            "historical_no_go_rewritten": False,
            "historical_formal_cell_data_access_allowed": False,
            "old_outcome_may_choose_margin": False,
            "old_outcome_may_choose_endpoint": False,
            "old_outcome_may_choose_family": False,
            "pilot_may_choose": [
                "round_sample_count",
                "trajectory_sample_count",
            ],
        },
        "G22_seven_metric_families": (
            snapshot.get("family_count") == 7
            and snapshot.get("family_names")
            == [
                "cutoff_mapping",
                "fault_trajectory_tail",
                "iq_conditional_distribution",
                "likelihood_score_posterior",
                "logical_ptm_survival",
                "physical_state_channel",
                "reset_leakage",
            ]
        ),
        "G23_every_metric_has_margin": (
            snapshot.get("all_family_metrics_have_margin") is True
            and snapshot.get("margin_count") == 27
        ),
        "G24_margin_provenance_and_domain": snapshot.get(
            "all_margins_positive_finite_with_source"
        )
        is True,
        "G25_all_cells_no_postselection": (
            snapshot.get("all_cells_required") is True
            and snapshot.get("no_postselection") is True
        ),
        "G26_cutoff_tail_ladder_frozen": (
            snapshot.get("cutoff_ladder") == [8, 12, 16, 20]
            and snapshot.get("primary_cutoff_increments")
            == [[12, 16], [16, 20]]
        ),
        "G27_global_iut_cell_tost": (
            snapshot.get("global_test") == "intersection_union_equivalence"
            and snapshot.get("cell_test") == "two_one_sided_tests"
        ),
        "G28_equivalence_alpha_frozen": (
            snapshot.get("cell_confidence_interval") == 0.90
            and snapshot.get("global_type_i_error") == 0.05
        ),
        "G29_mixed_unit_composite_forbidden": (
            snapshot.get("mixed_unit_vector_max") is False
        ),
        "G30_raw_log_evidence_diagnostic_only": snapshot.get(
            "raw_log_evidence_policy"
        )
        == "diagnostic only; never a cross-gain primary gate",
        "G31_no_cell_or_mean_rescue": (
            snapshot.get("cell_deletion") is False
            and snapshot.get("mean_only_rescue") is False
        ),
        "G32_nonfinite_fail_closed": snapshot.get("nonfinite_policy")
        == INCOMPLETE_FORMAL_VERDICT,
        "G33_seed_cluster_unit_frozen": snapshot.get("cluster_unit")
        == "independent seed position; all rows sharing a seed remain together",
        "G34_design_power_not_demo": (
            isinstance(snapshot.get("power_pseudoexperiments"), int)
            and snapshot["power_pseudoexperiments"] >= 4000
            and snapshot.get("power_formal_pool_access") is False
        ),
        "G35_outside_margin_false_pass_powered": (
            snapshot.get("power_false_pass_ucb") == 0.05
        ),
        "G36_same_backend_null_powered": (
            snapshot.get("power_same_a_lcb") == 0.90
            and snapshot.get("power_same_b_covered") is True
        ),
        "G37_ab_zero_effect_powered": snapshot.get("power_ab_lcb") == 0.90,
        "G38_predeclared_mutants_powered": (
            snapshot.get("power_mutants_covered") is True
        ),
        "G39_base_measure_frozen": snapshot.get("readout_base_measure")
        == "two_dimensional_lebesgue_per_complex_iq_sample",
        "G40_sigma_and_integration_frozen": (
            snapshot.get("sigma_convention")
            == "per_real_axis_standard_deviation"
            and snapshot.get("integration_convention")
            == "arithmetic_mean_over_window"
        ),
        "G41_latent_window_conditioning_frozen": snapshot.get(
            "latent_conditioning"
        )
        == "one_ancilla_label_per_complete_window",
        "G42_gain_jacobian_and_score_units_frozen": (
            snapshot.get("gain_jacobian_rule")
            == "-N*log(abs(det(G))) applied exactly once"
            and snapshot.get("proper_score_unit")
            == "nats_per_complex_sample"
        ),
        "G43_third_reference_runtime_independent": (
            snapshot.get("reference_no_backend_import") is True
            and snapshot.get("reference_no_random_or_numpy_import") is True
        ),
        "G44_third_reference_semantics_complete": snapshot.get(
            "reference_formula_tokens"
        )
        is True,
        "G45_runner_executes_independent_backends": (
            snapshot.get("runner_imports_backend_a") is True
            and snapshot.get("runner_imports_backend_b") is True
        ),
        "G46_runner_has_no_formal_cli_override": snapshot.get(
            "runner_no_cli_formal_overrides"
        )
        is True,
        "G47_runner_requires_exact_receipt": snapshot.get(
            "runner_receipt_exact_tokens"
        )
        is True,
        "G48_runner_attempt_accounting_complete": snapshot.get(
            "runner_complete_attempt_tokens"
        )
        is True,
        "G49_verifier_runtime_independent": snapshot.get(
            "verifier_no_physics_import"
        )
        is True,
        "G50_verifier_three_way_branch_frozen": snapshot.get(
            "verifier_three_verdict_tokens"
        )
        is True,
        "G51_fifteen_claims_and_scoped_claim": snapshot.get(
            "verifier_claim_tokens"
        )
        is True,
        "G52_downstream_release_scope_frozen": snapshot.get(
            "verifier_release_tokens"
        )
        is True,
        "G53_verifier_iut_tost_frozen": snapshot.get(
            "verifier_iut_tokens"
        )
        is True,
        "G54_verifier_not_outcome_hardcoded": snapshot.get(
            "verifier_no_outcome_hardcode"
        )
        is True,
        "G55_auditor_runtime_independent": snapshot.get(
            "auditor_no_runtime_import"
        )
        is True,
        "G56_outcome_blind_three_fixture_audit": (
            snapshot.get("auditor_outcome_blind_fixture_tokens") is True
            and snapshot.get("fixture_pass") is True
            and snapshot.get("fixture_no_go") is True
            and snapshot.get("fixture_incomplete") is True
        ),
        "G57_formal_outputs_and_attempts_absent": snapshot.get(
            "formal_existing"
        )
        == [],
        "G58_exact_preformal_git_commit": (
            isinstance(snapshot.get("commit_sha"), str)
            and bool(_COMMIT_RE.fullmatch(snapshot["commit_sha"]))
            and snapshot.get("commit_head_exact") is True
            and snapshot.get("commit_paths_exact") is True
            and isinstance(snapshot.get("protected_path_count"), int)
            and snapshot["protected_path_count"] >= 24
        ),
        "G59_fresh_qualification_binds_design_power": snapshot.get(
            "qualification_binds_design_report"
        )
        is True,
        "G60_verification_blueprint_exact": (
            snapshot.get("gate_blueprint_count") == 1589
            and snapshot.get("gate_blueprint_ids_unique") is True
            and snapshot.get("gate_blueprint_rows_complete") is True
            and snapshot.get("gate_blueprint_declared_sha256")
            == snapshot.get("gate_blueprint_sha256")
            and snapshot.get("gate_blueprint_declared_design_sha256")
            == snapshot.get("gate_blueprint_design_sha256")
        ),
        "G61_source_rows_not_duplicated": snapshot.get(
            "all_source_csv_rows_unique"
        )
        is True,
        "G62_source_tables_have_no_nonfinite_tokens": snapshot.get(
            "all_source_csv_no_nonfinite_tokens"
        )
        is True,
        "G63_verifier_rechecks_full_seal_policy": snapshot.get(
            "verifier_seal_policy_tokens"
        )
        is True,
        "G64_lineage_receipt_covers_all_fresh_sources": snapshot.get(
            "lineage_fresh_scan_live"
        )
        is True,
        "G65_gate_directions_and_scale_exact": (
            snapshot.get("gate_blueprint_directions_exact") is True
            and snapshot.get("gate_blueprint_normalized_sd_valid") is True
        ),
        "G66_gate_blueprint_metadata_exact": (
            snapshot.get("gate_blueprint_declared_row_count") == 1589
            and snapshot.get("gate_blueprint_source_config")
            == "configs/phase9/t_risk_20260726_01_design_power.json"
        ),
        "G67_verification_policy_exact": snapshot.get(
            "verification_contract_exact"
        )
        is True,
        "G68_fault_six_state_initialization_exact": snapshot.get(
            "fault_logical_initialization_exact"
        )
        is True,
        "G69_fault_logical_survival_not_permanent_null": (
            snapshot.get("runner_fault_logical_tokens") is True
            and snapshot.get(
                "verifier_terminal_logical_survival_token"
            )
            is True
        ),
        "G70_runtime_dependency_closure_exact": (
            snapshot.get("runtime_dependency_contract_exact") is True
            and snapshot.get("runtime_dependency_binding_count") == 11
        ),
    }


def _set_path(
    value: dict[str, Any],
    path: Sequence[str],
    replacement: object,
) -> None:
    cursor: dict[str, Any] = value
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"mutation path missing: {path}")
        cursor = child
    cursor[path[-1]] = replacement


def _snapshot_mutations() -> tuple[tuple[str, str, object], ...]:
    """Return one killable mutation per pre-formal gate.

    This intentionally exceeds the requested 48 mutations and includes
    scientific semantics, governance, power, source independence and release
    propagation rather than merely changing hashes.
    """

    return (
        ("M01_task_id", "task_id", "T9.2.4"),
        ("M02_lineage_verdict", "lineage_verdict", "PASS_REWRITTEN"),
        ("M03_rewrite_old_no_go", "historical_parent_rewritten", True),
        ("M04_old_cell_access", "fresh_old_cell_reference_count", 1),
        ("M05_diagnostic_fail", "diagnostic_verdict", "INCOMPLETE_FAIL_CLOSED"),
        ("M06_old_margin_selection", "diagnostic_old_outcome_selected_margin", True),
        ("M07_readout_fail", "readout_verdict", "NO_GO"),
        ("M08_readout_formal_access", "readout_formal_pool_accessed", True),
        ("M09_design_fail", "design_verdict", "NO_GO"),
        ("M10_design_old_outcome", "design_old_outcome_used", True),
        ("M11_config_hash_drift", "config_hash_matches_design", False),
        ("M12_missing_source_row", "all_source_csv_nonempty_rectangular", False),
        ("M13_missing_seed_split", "split_count", 15),
        ("M14_seed_overlap", "splits_disjoint", False),
        ("M15_historical_seed_namespace", "fresh_seed_namespace", False),
        ("M16_round_seed_change", "formal_round_pool", ((1, 769), (2, 770))),
        ("M17_trajectory_seed_change", "formal_trajectory_pool", ((1, 257), (2, 258))),
        ("M18_underpowered_formal", "selected_round_count", 128),
        ("M19_candidate_change", "candidate_round_counts", [16, 32]),
        ("M20_pilot_changes_margin", "pilot_allowed_fields", ["margin"]),
        ("M21_old_endpoint_selection", "old_outcome_policy", {}),
        ("M22_drop_family", "family_count", 6),
        ("M23_metric_without_margin", "all_family_metrics_have_margin", False),
        ("M24_nan_margin", "all_margins_positive_finite_with_source", False),
        ("M25_postselection", "no_postselection", False),
        ("M26_drop_cutoff_tail", "cutoff_ladder", [8, 12]),
        ("M27_replace_iut", "global_test", "aggregate_mean"),
        ("M28_wrong_alpha", "global_type_i_error", 0.10),
        ("M29_mixed_unit_max", "mixed_unit_vector_max", True),
        ("M30_raw_score_primary", "raw_log_evidence_policy", "primary"),
        ("M31_cell_deletion", "cell_deletion", True),
        ("M32_nan_to_zero", "nonfinite_policy", NO_GO_FORMAL_VERDICT),
        ("M33_row_resampling", "cluster_unit", "row"),
        ("M34_demo_power", "power_pseudoexperiments", 20),
        ("M35_outside_margin_weak", "power_false_pass_ucb", 0.20),
        ("M36_missing_b_null", "power_same_b_covered", False),
        ("M37_missing_ab_pilot", "power_ab_lcb", 0.0),
        ("M38_alternative_below_margin", "power_mutants_covered", False),
        ("M39_wrong_base_measure", "readout_base_measure", "one_dimensional"),
        ("M40_sigma_as_variance", "sigma_convention", "variance"),
        ("M41_redraw_latent_each_sample", "latent_conditioning", "per_sample"),
        ("M42_omit_gain_jacobian", "gain_jacobian_rule", "omitted"),
        ("M43_shared_backend_oracle", "reference_no_backend_import", False),
        ("M44_factor_two_formula", "reference_formula_tokens", False),
        ("M45_drop_backend_b", "runner_imports_backend_b", False),
        ("M46_cli_threshold_override", "runner_no_cli_formal_overrides", False),
        ("M47_arbitrary_receipt", "runner_receipt_exact_tokens", False),
        ("M48_drop_attempt_rows", "runner_complete_attempt_tokens", False),
        ("M49_verifier_imports_physics", "verifier_no_physics_import", False),
        ("M50_branch_swap", "verifier_three_verdict_tokens", False),
        ("M51_promote_claim", "verifier_claim_tokens", False),
        ("M52_release_propagation", "verifier_release_tokens", False),
        ("M53_wrong_bound_direction", "verifier_iut_tokens", False),
        ("M54_postaudit_outcome_hardcode", "verifier_no_outcome_hardcode", False),
        ("M55_auditor_imports_runner", "auditor_no_runtime_import", False),
        ("M56_missing_incomplete_fixture", "fixture_incomplete", False),
        ("M57_formal_output_exists", "formal_existing", ["attempt.csv"]),
        ("M58_uncommitted_source", "commit_paths_exact", False),
        ("M59_unbound_design_report", "qualification_binds_design_report", False),
        ("M60_gate_blueprint_drift", "gate_blueprint_count", 1588),
        ("M61_gate_blueprint_hash", "gate_blueprint_declared_sha256", "0" * 64),
        (
            "M62_design_blueprint_hash",
            "gate_blueprint_declared_design_sha256",
            "0" * 64,
        ),
        ("M63_duplicate_source_row", "all_source_csv_rows_unique", False),
        (
            "M64_nan_source_value",
            "all_source_csv_no_nonfinite_tokens",
            False,
        ),
        (
            "M65_weak_verifier_seal_policy",
            "verifier_seal_policy_tokens",
            False,
        ),
        ("M66_stale_lineage_scan", "lineage_fresh_scan_live", False),
        (
            "M67_wrong_principal_direction",
            "gate_blueprint_directions_exact",
            False,
        ),
        (
            "M68_blueprint_metadata_drift",
            "gate_blueprint_declared_row_count",
            1588,
        ),
        (
            "M69_verification_policy_drift",
            "verification_contract_exact",
            False,
        ),
        (
            "M70_fault_vacuum_initialization",
            "fault_logical_initialization_exact",
            False,
        ),
        (
            "M71_fault_survival_permanent_null",
            "runner_fault_logical_tokens",
            False,
        ),
        (
            "M72_unbound_runtime_dependency",
            "runtime_dependency_binding_count",
            10,
        ),
    )


def run_mutation_audit(snapshot: Mapping[str, Any]) -> list[dict[str, object]]:
    baseline = audit_snapshot(snapshot)
    if not all(baseline.values()):
        failed = [name for name, passed in baseline.items() if not passed]
        raise ValueError(f"baseline preformal gates failed: {failed}")
    results: list[dict[str, object]] = []
    for mutation_id, field, replacement in _snapshot_mutations():
        mutated = copy.deepcopy(dict(snapshot))
        mutated[field] = replacement
        gates = audit_snapshot(mutated)
        detected = not all(gates.values())
        results.append(
            {
                "mutation_id": mutation_id,
                "field": field,
                "detected": detected,
                "failed_gates": [
                    name for name, passed in gates.items() if not passed
                ],
            }
        )
    return results


def _fixture_mutations() -> list[dict[str, object]]:
    cases: tuple[
        tuple[str, str, str, object],
        ...,
    ] = (
        ("F-M01_pass_claim_null", "pass", "qualified_claim", None),
        ("F-M02_pass_partial_release", "pass", "released_tasks", ["T9.2.5"]),
        ("F-M03_pass_blocked", "pass", "blocked_tasks", ["T9.6.5"]),
        ("F-M04_pass_gate_fail", "pass", "all_main_gates_passed", False),
        ("F-M05_pass_infra_error", "pass", "infrastructure_errors", ["x"]),
        ("F-M06_no_go_claim", "no_go", "qualified_claim", QUALIFIED_CLAIM),
        ("F-M07_no_go_release", "no_go", "released_tasks", list(DOWNSTREAM_TASKS)),
        ("F-M08_no_go_drop_block", "no_go", "blocked_tasks", []),
        ("F-M09_no_go_incomplete", "no_go", "complete_finite_denominator", False),
        ("F-M10_incomplete_no_error", "incomplete", "infrastructure_errors", []),
        ("F-M11_incomplete_release", "incomplete", "released_tasks", ["T9.2.5"]),
        ("F-M12_incomplete_claim", "incomplete", "qualified_claim", QUALIFIED_CLAIM),
        ("F-M13_incomplete_complete", "incomplete", "complete_finite_denominator", True),
        ("F-M14_unknown_branch", "pass", "verdict", "PASS"),
        ("F-M15_claim_false", "pass", "claim_state", {**{field: None for field in CLAIM_FIELDS}, "rank": False}),
        ("F-M16_claim_missing", "no_go", "claim_state", {}),
        ("F-M17_claim_extra", "incomplete", "claim_state", {**{field: None for field in CLAIM_FIELDS}, "surpass": None}),
        ("F-M18_outcome_hardcode", "pass", "outcome_specific_audit_rule", "expect PASS"),
    )
    results: list[dict[str, object]] = []
    for mutation_id, branch, field, replacement in cases:
        fixture = _formal_fixture(branch)
        fixture[field] = replacement
        detected = not all(audit_outcome_fixture(fixture).values())
        results.append(
            {
                "mutation_id": mutation_id,
                "branch": branch,
                "field": field,
                "detected": detected,
            }
        )
    return results


def build_audit(
    root: Path,
    *,
    preformal_commit_sha: str,
) -> dict[str, Any]:
    snapshot = build_snapshot(
        root,
        preformal_commit_sha=preformal_commit_sha,
    )
    gates = audit_snapshot(snapshot)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"preformal audit gates failed: {failed}")
    mutations = run_mutation_audit(snapshot)
    fixture_mutations = _fixture_mutations()
    if not all(item["detected"] is True for item in mutations):
        raise ValueError("preformal mutation audit did not kill every mutant")
    if not all(item["detected"] is True for item in fixture_mutations):
        raise ValueError("outcome fixture mutation audit incomplete")

    root = root.resolve()
    inputs = {
        name: _binding(root, relative)
        for name, relative in INPUT_PATHS.items()
    }
    sources = _all_source_bindings(root)
    tests = {
        name: _binding(root, relative)
        for name, relative in TEST_PATHS.items()
    }
    fixtures = {
        branch: _formal_fixture(branch)
        for branch in ("pass", "no_go", "incomplete")
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS_FRESH_TWIN_PREFORMAL_AUDIT",
        "verdict": PASS_VERDICT,
        "formal_result_accessed": False,
        "preformal_commit_sha": preformal_commit_sha,
        "input_bindings": inputs,
        "source_bindings": sources,
        "test_bindings": tests,
        "formal_output_paths": list(FORMAL_OUTPUT_PATHS),
        "formal_outputs_existing": [],
        "historical_formal_cell_data_accessed": False,
        "scientific_verdict": None,
        "selected_sample_counts": {
            "round_sample_count": snapshot["selected_round_count"],
            "trajectory_sample_count": snapshot[
                "selected_trajectory_count"
            ],
        },
        "statistical_contract": {
            "global_test": snapshot["global_test"],
            "cell_test": snapshot["cell_test"],
            "cell_confidence_interval": snapshot[
                "cell_confidence_interval"
            ],
            "global_type_i_error": snapshot["global_type_i_error"],
            "cluster_unit": snapshot["cluster_unit"],
            "missing_nonfinite_exception": snapshot["nonfinite_policy"],
            "raw_log_evidence_primary": False,
            "mixed_unit_vector_max": False,
            "cell_deletion": False,
            "mean_only_rescue": False,
        },
        "verification_contract": {
            "gate_blueprint_count": snapshot["gate_blueprint_count"],
            "blueprint_sha256": snapshot[
                "gate_blueprint_sha256"
            ],
            "independent_verifier": sources["formal_verifier"],
        },
        "outcome_blind_fixtures": fixtures,
        "formal_verdicts": {
            "pass": PASS_FORMAL_VERDICT,
            "scientific_no_go": NO_GO_FORMAL_VERDICT,
            "incomplete": INCOMPLETE_FORMAL_VERDICT,
        },
        "qualified_claim_if_and_only_if_pass": QUALIFIED_CLAIM,
        "claim_state": {field: None for field in CLAIM_FIELDS},
        "released_if_and_only_if_pass": list(DOWNSTREAM_TASKS),
        "gates": gates,
        "all_gates_passed": True,
        "gate_summary": {
            "passed": sum(value is True for value in gates.values()),
            "total": len(gates),
            "all_passed": all(gates.values()),
        },
        "semantic_mutations": mutations,
        "fixture_mutations": fixture_mutations,
        "mutation_summary": {
            "detected": sum(
                item["detected"] is True
                for item in [*mutations, *fixture_mutations]
            ),
            "total": len(mutations) + len(fixture_mutations),
            "all_detected": all(
                item["detected"] is True
                for item in [*mutations, *fixture_mutations]
            ),
        },
        "all_mutations_detected": True,
        "claim_boundary": (
            "This seal qualifies no scientific or hardware result. It only "
            "authorizes one fresh formal transaction at the exact bound "
            "implementation/configuration. All fifteen performance and "
            "external-comparison fields remain literal null."
        ),
    }
    report["analysis_sha256"] = _sha(report)
    return report


def build_seal(
    root: Path,
    *,
    preformal_commit_sha: str,
    audit_path: Path | None = None,
    audit_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind an already persisted, exactly regenerated pre-formal audit."""

    root = root.resolve()
    expected_audit = build_audit(
        root,
        preformal_commit_sha=preformal_commit_sha,
    )
    if audit_path is None:
        qualification = _load_json(
            root / INPUT_PATHS["qualification_config"]
        )
        relative = qualification.get("preformal_seal", {}).get(
            "audit_path"
        )
        if not isinstance(relative, str):
            raise ValueError("qualification config lacks preformal audit path")
        audit_path = root / relative
    else:
        audit_path = (
            audit_path
            if audit_path.is_absolute()
            else root / audit_path
        )
    if audit_document is not None:
        persisted_audit = dict(audit_document)
    elif audit_path.exists():
        persisted_audit = _load_json(audit_path)
    else:
        raise ValueError("persisted preformal audit is missing")
    if _canonical(persisted_audit) != _canonical(expected_audit):
        raise ValueError(
            "persisted preformal audit is missing, stale, or not exact"
        )
    audit_relative = audit_path.resolve().relative_to(root).as_posix()
    if audit_path.exists():
        audit_binding = _binding(root, audit_relative)
    else:
        payload = _serialized(persisted_audit)
        audit_binding = {
            "path": audit_relative,
            "bytes": len(payload),
            "sha256": _sha_bytes(payload),
        }
    inputs = expected_audit["input_bindings"]
    sources = expected_audit["source_bindings"]
    seal: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "verdict": PASS_VERDICT,
        "formal_result_accessed": False,
        "historical_formal_cell_data_accessed": False,
        "scientific_verdict": None,
        "preformal_commit_sha": preformal_commit_sha,
        "preformal_audit_analysis_sha256": persisted_audit[
            "analysis_sha256"
        ],
        "live_bindings": {
            "fresh_config": inputs["qualification_config"],
            "fresh_runner": sources["formal_runner"],
            "historical_lineage_receipt": inputs["historical_lineage"],
            "design_power_report": inputs["design_power"],
            "preformal_audit": audit_binding,
            **{
                name: binding
                for name, binding in sources.items()
                if name.startswith("runtime_dependency_")
            },
        },
        "input_bindings": inputs,
        "source_bindings": sources,
        "test_bindings": expected_audit["test_bindings"],
        "formal_output_paths": expected_audit["formal_output_paths"],
        "formal_outputs_existing": [],
        "selected_sample_counts": expected_audit[
            "selected_sample_counts"
        ],
        "statistical_contract": expected_audit[
            "statistical_contract"
        ],
        "verification_contract": expected_audit[
            "verification_contract"
        ],
        "outcome_blind_fixtures": expected_audit[
            "outcome_blind_fixtures"
        ],
        "formal_verdicts": expected_audit["formal_verdicts"],
        "qualified_claim_if_and_only_if_pass": QUALIFIED_CLAIM,
        "claim_state": {field: None for field in CLAIM_FIELDS},
        "released_if_and_only_if_pass": list(DOWNSTREAM_TASKS),
        "gates": expected_audit["gates"],
        "all_gates_passed": True,
        "gate_summary": expected_audit["gate_summary"],
        "semantic_mutations": expected_audit["semantic_mutations"],
        "fixture_mutations": expected_audit["fixture_mutations"],
        "mutation_summary": expected_audit["mutation_summary"],
        "all_mutations_detected": True,
        "claim_boundary": expected_audit["claim_boundary"],
    }
    seal["analysis_sha256"] = _sha(seal)
    return seal


def verify_seal(
    seal: Mapping[str, Any],
    root: Path,
    *,
    require_outputs_absent: bool = True,
) -> bool:
    """Verify exact receipt bytes and optionally preserve one-shot absence."""

    claimed = seal.get("analysis_sha256")
    if not isinstance(claimed, str):
        return False
    payload = dict(seal)
    payload.pop("analysis_sha256", None)
    if _sha(payload) != claimed:
        return False
    if seal.get("schema_version") != SCHEMA_VERSION:
        return False
    if seal.get("verdict") != PASS_VERDICT:
        return False
    live = seal.get("live_bindings")
    required_live = {
        "fresh_config",
        "fresh_runner",
        "historical_lineage_receipt",
        "design_power_report",
        "preformal_audit",
        *{
            f"runtime_dependency_{index:02d}"
            for index in range(len(RUNTIME_DEPENDENCY_PATHS))
        },
    }
    if not isinstance(live, dict) or set(live) != required_live:
        return False
    for binding in live.values():
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or _binding(root.resolve(), binding["path"]) != binding
        ):
            return False
    for index, expected_path in enumerate(RUNTIME_DEPENDENCY_PATHS):
        runtime_binding = live[f"runtime_dependency_{index:02d}"]
        if runtime_binding.get("path") != expected_path:
            return False
    preaudit_binding = live["preformal_audit"]
    preaudit = _load_json(root.resolve() / preaudit_binding["path"])
    if (
        not _analysis_hash_valid(preaudit)
        or preaudit.get("analysis_sha256")
        != seal.get("preformal_audit_analysis_sha256")
    ):
        return False
    for section in ("input_bindings", "source_bindings", "test_bindings"):
        bindings = seal.get(section)
        if not isinstance(bindings, dict):
            return False
        for binding in bindings.values():
            if (
                not isinstance(binding, dict)
                or not isinstance(binding.get("path"), str)
                or _binding(root.resolve(), binding["path"]) != binding
            ):
                return False
    if require_outputs_absent and any(
        (root / relative).exists() for relative in FORMAL_OUTPUT_PATHS
    ):
        return False
    return True


def _write_one_shot(
    path: Path,
    document: Mapping[str, Any],
    *,
    artifact_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"{artifact_name} is one-shot and already exists: {path}"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_serialized(document).decode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_audit(path: Path, report: Mapping[str, Any]) -> None:
    _write_one_shot(
        path,
        report,
        artifact_name="preformal audit",
    )


def write_seal(path: Path, seal: Mapping[str, Any]) -> None:
    _write_one_shot(
        path,
        seal,
        artifact_name="preformal seal",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preformal-commit",
        required=True,
        help="exact lowercase 40-hex commit containing every bound input",
    )
    parser.add_argument("--root", type=Path, default=_root())
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "docs/t_risk_20260726_01_fresh_preformal_audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/t_risk_20260726_01_fresh_preformal_seal.json"
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    audit_output = (
        args.audit_output
        if args.audit_output.is_absolute()
        else root / args.audit_output
    )
    output = args.output if args.output.is_absolute() else root / args.output
    report = build_audit(
        root,
        preformal_commit_sha=args.preformal_commit,
    )
    seal = build_seal(
        root,
        preformal_commit_sha=args.preformal_commit,
        audit_path=audit_output,
        audit_document=report,
    )
    write_audit(audit_output, report)
    write_seal(output, seal)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "verdict": seal["verdict"],
                "audit_analysis_sha256": report["analysis_sha256"],
                "analysis_sha256": seal["analysis_sha256"],
                "gate_summary": seal["gate_summary"],
                "mutation_summary": seal["mutation_summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CLAIM_FIELDS",
    "DOWNSTREAM_TASKS",
    "FORMAL_OUTPUT_PATHS",
    "INCOMPLETE_FORMAL_VERDICT",
    "INPUT_PATHS",
    "NO_GO_FORMAL_VERDICT",
    "PASS_FORMAL_VERDICT",
    "PASS_VERDICT",
    "QUALIFIED_CLAIM",
    "RUNTIME_DEPENDENCY_PATHS",
    "SCHEMA_VERSION",
    "SOURCE_PATHS",
    "STATUS",
    "TASK_ID",
    "TEST_PATHS",
    "audit_outcome_fixture",
    "audit_snapshot",
    "build_audit",
    "build_seal",
    "build_snapshot",
    "run_mutation_audit",
    "verify_seal",
    "write_audit",
    "write_seal",
]
