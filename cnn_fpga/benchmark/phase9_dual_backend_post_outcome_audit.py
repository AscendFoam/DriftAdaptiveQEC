"""Independent post-outcome audit for the frozen T9.2.4 qualification.

This module deliberately imports neither physics implementation nor the formal
verifier.  It checks the immutable evidence chain, reconstructs the worst
point estimate directly from the row ledger, and confirms that the scientific
NO-GO propagates without promoting any performance claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TASK_ID = "T9.2.4"
SCHEMA = "PHASE9-DUAL-BACKEND-POST-OUTCOME-AUDIT-V1"
VERDICT = "AUDIT_CONFIRMS_SCIENTIFIC_NO_GO"
EXPECTED_REPORT_ANALYSIS = (
    "d3f05e69db9c4799d55ce0813f8bd52cdb22384dbd5cfc68227715c44c53ec49"
)
EXPECTED_PARENT_ANALYSIS = (
    "98e72c457dab941daab270b3bd63eec939564a6f1fedaad061eab59280988695"
)
EXPECTED_CHILD_ANALYSIS = (
    "f2a2ab287c59e0947768cd08824856d7b9046b79c3d713b18043efe09de22944"
)
EXPECTED_EXECUTION = (
    "96ee4483189ecd15c63a8eb9a146f52f3d91166bc9a5cf1d5310fdd25b3665ff"
)
EXPECTED_CLAIMS = (
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "hardware_measured",
    "rank",
)
EXPECTED_RELEASED = ("T9.2.6",)
EXPECTED_BLOCKED = (
    "T9.2.5",
    "T9.2.7",
    "T9.3.1",
    "T9.3.4",
    "T9.6.2",
    "T9.6.5",
)
EXPECTED_FAILURES = {
    "ensemble_density_trace_distance": 2,
    "integrated_iq_mean_difference": 66,
    "iq_two_sample_ks": 66,
    "leakage_residence_rate_difference": 2,
    "level_probability_l1": 3,
    "log_evidence_mean_difference": 66,
    "logical_survival_difference": 2,
    "reset_success_rate_difference": 2,
    "short_trajectory_observable_mean_difference": 8,
}
WORST_GATE = "ab|probe|P07_BOUNDARY|XZ|cutoff=8|log_evidence"
WORST_CELL = "probe|P07_BOUNDARY|XZ|cutoff=8"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _binding(root: Path, relative: str) -> dict[str, object]:
    payload = (root / relative).read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _report_self_hash(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("analysis_sha256", None)
    return _sha_bytes(_canonical(payload).encode("utf-8"))


def _all_release_bindings_match(
    root: Path,
    release: Mapping[str, Any],
) -> bool:
    for section in ("lineage", "implementation", "evidence"):
        entries = release.get(section)
        if not isinstance(entries, Mapping):
            return False
        for binding in entries.values():
            if not isinstance(binding, Mapping):
                return False
            relative = binding.get("path")
            if not isinstance(relative, str):
                return False
            actual = _binding(root, relative)
            if actual != dict(binding):
                return False
    return True


def _failure_counts(gates: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in gates:
        if row["passed"].lower() != "false":
            continue
        metric = row["metric"]
        counts[metric] = counts.get(metric, 0) + 1
    return dict(sorted(counts.items()))


def build_snapshot(root: Path) -> dict[str, Any]:
    report_path = root / "docs/t9_2_4_dual_backend_qualification.json"
    manifest_path = root / "docs/t9_2_4_dual_backend_execution_manifest.json"
    source_path = (
        root / "docs/t9_2_4_dual_backend_qualification_source_data.csv"
    )
    ledger_path = root / "docs/t9_2_4_dual_backend_cell_ledger.csv"
    archive_path = root / "docs/t9_2_4_dual_backend_state_archive.npz"
    release_path = root / "configs/phase9/t9_2_4_release_pin.json"

    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    release = _load_json(release_path)
    source_rows = _read_csv(source_path)
    ledger = _read_csv(ledger_path)
    gates = [
        row for row in source_rows if row["record_type"] == "gate_metric"
    ]
    mapping_rows = [
        row
        for row in source_rows
        if row["record_type"] == "mapping_matrix_entry"
    ]
    failed = [row for row in gates if row["passed"].lower() == "false"]
    passed = [row for row in gates if row["passed"].lower() == "true"]
    worst = min(gates, key=lambda row: float(row["margin"]))
    worst_rows = [row for row in ledger if row["cell_id"] == WORST_CELL]
    worst_a = [
        float(row["log_evidence"])
        for row in worst_rows
        if row["backend"] == "A"
    ]
    worst_b = [
        float(row["log_evidence"])
        for row in worst_rows
        if row["backend"] == "B"
    ]
    independent_point = abs(float(np.mean(worst_a)) - float(np.mean(worst_b)))

    seed_a = {int(row["seed"]) for row in ledger if row["backend"] == "A"}
    seed_b = {int(row["seed"]) for row in ledger if row["backend"] == "B"}
    ledger_row_ids = [row["row_id"] for row in ledger]
    expected_archive_ids = {
        row["row_id"]
        for row in ledger
        if row["layer"] != "fault" or row["terminal_round"] == "True"
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        archive_schema = archive["schema"].tolist()
        archive_ids = {
            str(value)
            for cutoff in (8, 12)
            for value in archive[f"row_ids_cutoff_{cutoff}"].tolist()
        }
        archive_shapes = {
            name: list(archive[name].shape) for name in archive.files
        }
        captured = {
            f"{backend}_{cutoff}": archive[
                f"mapping_captured_{backend}_cutoff_{cutoff}"
            ].astype(float).tolist()
            for backend in ("a", "b")
            for cutoff in (8, 12)
        }

    target = next(row for row in gates if row["gate_id"] == WORST_GATE)
    target_bound_recomputed = float(target["estimate"]) + float(
        report["bootstrap"]["critical_value"]
    ) * float(target["standard_error"])
    report_claims = report.get("claim_state", {})
    release_claims = release.get("claim_state", {})
    return {
        "report_analysis": report.get("analysis_sha256"),
        "report_self_hash": _report_self_hash(report),
        "report_verdict": report.get("verdict"),
        "report_qualified_claim": report.get("qualified_claim"),
        "parent_analysis": report.get("lineage", {}).get(
            "parent_preregistration_analysis_sha256"
        ),
        "child_analysis": report.get("lineage", {}).get(
            "formal_child_seal_analysis_sha256"
        ),
        "execution_analysis": report.get("lineage", {}).get(
            "execution_sha256"
        ),
        "manifest_execution_analysis": manifest.get("execution_sha256"),
        "manifest_expected": manifest.get("expected_rows"),
        "manifest_observed": manifest.get("observed_rows"),
        "manifest_exceptions": manifest.get("exception_rows"),
        "manifest_conservation": manifest.get("conservation_pass_rows"),
        "ledger_count": len(ledger),
        "ledger_unique": len(set(ledger_row_ids)),
        "ledger_exception_count": sum(
            bool(row["exception_type"]) for row in ledger
        ),
        "ledger_conservation_count": sum(
            row["conservation_pass"] == "True" for row in ledger
        ),
        "seed_a_count": len(seed_a),
        "seed_b_count": len(seed_b),
        "seed_overlap": len(seed_a & seed_b),
        "archive_schema": archive_schema,
        "archive_ids_count": len(archive_ids),
        "archive_expected_count": len(expected_archive_ids),
        "archive_id_coverage": archive_ids == expected_archive_ids,
        "archive_shapes": archive_shapes,
        "captured": captured,
        "source_count": len(source_rows),
        "source_gate_count": len(gates),
        "source_mapping_count": len(mapping_rows),
        "source_passed_count": len(passed),
        "source_failed_count": len(failed),
        "failure_counts": _failure_counts(gates),
        "all_gate_notes_no_postselection": all(
            row["notes"] == "trace-decreasing/no-postselection where logical"
            for row in gates
        ),
        "report_gate_summary": report.get("gate_summary"),
        "worst_gate": worst["gate_id"],
        "worst_margin": float(worst["margin"]),
        "report_worst_gate": report.get("worst_gate", {}).get("gate_id"),
        "report_worst_margin": report.get("worst_gate", {}).get("margin"),
        "worst_sample_count_a": len(worst_a),
        "worst_sample_count_b": len(worst_b),
        "worst_independent_point": independent_point,
        "worst_source_point": float(target["estimate"]),
        "worst_source_bound": float(target["simultaneous_bound"]),
        "worst_recomputed_bound": target_bound_recomputed,
        "worst_tolerance": float(target["tolerance"]),
        "bootstrap_resamples": report.get("bootstrap", {}).get("resamples"),
        "bootstrap_metric_count": report.get("bootstrap", {}).get(
            "total_metric_count"
        ),
        "infrastructure_errors": report.get("infrastructure_errors"),
        "released_tasks": sorted(
            report.get("failure_propagation", {}).get("released_tasks", [])
        ),
        "blocked_tasks": sorted(
            report.get("failure_propagation", {}).get("blocked_tasks", [])
        ),
        "report_claim_keys": sorted(report_claims),
        "report_claim_values": [report_claims.get(key) for key in EXPECTED_CLAIMS],
        "release_claim_keys": sorted(release_claims),
        "release_claim_values": [
            release_claims.get(key) for key in EXPECTED_CLAIMS
        ],
        "release_analysis": release.get("analysis_sha256"),
        "release_verdict": release.get("verdict"),
        "release_bindings_match": _all_release_bindings_match(root, release),
    }


def audit_snapshot(snapshot: Mapping[str, Any]) -> dict[str, bool]:
    captured = snapshot["captured"]
    gates = {
        "report_self_hash": (
            snapshot["report_analysis"] == EXPECTED_REPORT_ANALYSIS
            and snapshot["report_self_hash"] == EXPECTED_REPORT_ANALYSIS
        ),
        "lineage_exact": (
            snapshot["parent_analysis"] == EXPECTED_PARENT_ANALYSIS
            and snapshot["child_analysis"] == EXPECTED_CHILD_ANALYSIS
            and snapshot["execution_analysis"] == EXPECTED_EXECUTION
            and snapshot["manifest_execution_analysis"] == EXPECTED_EXECUTION
        ),
        "execution_row_total": (
            snapshot["manifest_expected"]
            == snapshot["manifest_observed"]
            == snapshot["ledger_count"]
            == snapshot["ledger_unique"]
            == 16800
        ),
        "zero_exception_rows": (
            snapshot["manifest_exceptions"]
            == snapshot["ledger_exception_count"]
            == 0
        ),
        "all_conservation_rows": (
            snapshot["manifest_conservation"]
            == snapshot["ledger_conservation_count"]
            == 16800
        ),
        "backend_seed_disjoint": (
            snapshot["seed_a_count"] == 112
            and snapshot["seed_b_count"] == 112
            and snapshot["seed_overlap"] == 0
        ),
        "archive_schema": snapshot["archive_schema"]
        == ["PHASE9-DUAL-BACKEND-STATE-ARCHIVE-V1"],
        "archive_terminal_coverage": (
            snapshot["archive_id_coverage"] is True
            and snapshot["archive_ids_count"]
            == snapshot["archive_expected_count"]
            == 11872
        ),
        "archive_density_shapes": (
            snapshot["archive_shapes"].get("densities_cutoff_8")
            == [10752, 24, 24]
            and snapshot["archive_shapes"].get("densities_cutoff_12")
            == [1120, 36, 36]
        ),
        "mapping_backend_alignment": all(
            np.allclose(
                captured[f"a_{cutoff}"],
                captured[f"b_{cutoff}"],
                atol=1e-14,
                rtol=0.0,
            )
            for cutoff in (8, 12)
        ),
        "mapping_cutoff_improves_min_capture": min(captured["a_12"])
        > min(captured["a_8"]),
        "source_record_total": (
            snapshot["source_count"] == 1538
            and snapshot["source_gate_count"] == 1042
            and snapshot["source_mapping_count"] == 496
        ),
        "gate_count_recomputed": (
            snapshot["source_passed_count"] == 825
            and snapshot["source_failed_count"] == 217
            and snapshot["report_gate_summary"]
            == {
                "passed": 825,
                "failed": 217,
                "total": 1042,
                "all_passed": False,
            }
        ),
        "failure_distribution_exact": snapshot["failure_counts"]
        == EXPECTED_FAILURES,
        "worst_gate_recomputed": (
            snapshot["worst_gate"] == WORST_GATE
            and snapshot["report_worst_gate"] == WORST_GATE
            and np.isclose(
                snapshot["worst_margin"],
                snapshot["report_worst_margin"],
                atol=1e-15,
                rtol=0.0,
            )
        ),
        "worst_point_from_ledger": (
            snapshot["worst_sample_count_a"] == 16
            and snapshot["worst_sample_count_b"] == 16
            and np.isclose(
                snapshot["worst_independent_point"],
                snapshot["worst_source_point"],
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "simultaneous_bound_arithmetic": np.isclose(
            snapshot["worst_source_bound"],
            snapshot["worst_recomputed_bound"],
            atol=1e-12,
            rtol=0.0,
        ),
        "worst_gate_fails_frozen_tolerance": snapshot["worst_source_bound"]
        > snapshot["worst_tolerance"],
        "bootstrap_scope": (
            snapshot["bootstrap_resamples"] == 2000
            and snapshot["bootstrap_metric_count"] == 1042
        ),
        "no_postselection_contract": snapshot[
            "all_gate_notes_no_postselection"
        ]
        is True,
        "scientific_not_infrastructure_failure": (
            snapshot["report_verdict"] == "NO_GO_TWIN_QUALIFICATION"
            and snapshot["report_qualified_claim"] is None
            and snapshot["infrastructure_errors"] == []
        ),
        "failure_propagation": (
            snapshot["released_tasks"] == sorted(EXPECTED_RELEASED)
            and snapshot["blocked_tasks"] == sorted(EXPECTED_BLOCKED)
        ),
        "all_claims_typed_null": (
            snapshot["report_claim_keys"] == sorted(EXPECTED_CLAIMS)
            and snapshot["release_claim_keys"] == sorted(EXPECTED_CLAIMS)
            and all(value is None for value in snapshot["report_claim_values"])
            and all(value is None for value in snapshot["release_claim_values"])
        ),
        "release_pin_matches_report": (
            snapshot["release_analysis"] == EXPECTED_REPORT_ANALYSIS
            and snapshot["release_verdict"] == "NO_GO_TWIN_QUALIFICATION"
        ),
        "release_file_bindings": snapshot["release_bindings_match"] is True,
    }
    return gates


def _mutation_audit(snapshot: Mapping[str, Any]) -> list[dict[str, object]]:
    cases: list[tuple[str, str, object]] = [
        ("report_hash", "report_self_hash", "0" * 64),
        ("parent_lineage", "parent_analysis", "0" * 64),
        ("execution_lineage", "execution_analysis", "0" * 64),
        ("drop_row", "ledger_count", 16799),
        ("duplicate_row", "ledger_unique", 16799),
        ("hide_exception", "ledger_exception_count", 1),
        ("hide_conservation_failure", "ledger_conservation_count", 16799),
        ("overlap_rng", "seed_overlap", 1),
        ("drop_archive_density", "archive_ids_count", 11871),
        ("fake_archive_coverage", "archive_id_coverage", False),
        ("drop_source_gate", "source_gate_count", 1041),
        ("promote_failed_gate", "source_failed_count", 216),
        ("rewrite_failure_family", "failure_counts", {}),
        ("replace_worst_gate", "worst_gate", "benign"),
        ("rewrite_point", "worst_source_point", 0.0),
        ("rewrite_bound", "worst_source_bound", 0.0),
        ("reduce_bootstrap", "bootstrap_resamples", 200),
        ("permit_postselection", "all_gate_notes_no_postselection", False),
        ("promote_verdict", "report_verdict", "PASS_TWIN_QUALIFICATION"),
        ("release_surrogate", "released_tasks", ["T9.2.5", "T9.2.6"]),
        ("drop_blocked_task", "blocked_tasks", []),
        ("promote_claim", "report_claim_values", [0.0] + [None] * 7),
        ("change_release_verdict", "release_verdict", "PASS_TWIN_QUALIFICATION"),
        ("tamper_bound_file", "release_bindings_match", False),
    ]
    results: list[dict[str, object]] = []
    baseline = audit_snapshot(snapshot)
    if not all(baseline.values()):
        raise ValueError("baseline post-outcome audit failed")
    for mutation_id, field, value in cases:
        mutated = copy.deepcopy(dict(snapshot))
        mutated[field] = value
        detected = not all(audit_snapshot(mutated).values())
        results.append(
            {
                "mutation_id": mutation_id,
                "field": field,
                "detected": detected,
            }
        )
    return results


def build_audit(root: Path) -> dict[str, Any]:
    snapshot = build_snapshot(root)
    gates = {
        name: bool(passed)
        for name, passed in audit_snapshot(snapshot).items()
    }
    mutations = _mutation_audit(snapshot)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"post-outcome audit gates failed: {failed}")
    if not all(item["detected"] is True for item in mutations):
        raise ValueError("post-outcome mutation audit incomplete")
    audit: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "verdict": VERDICT,
        "formal_report_analysis_sha256": EXPECTED_REPORT_ANALYSIS,
        "evidence": {
            "report": _binding(
                root, "docs/t9_2_4_dual_backend_qualification.json"
            ),
            "execution_manifest": _binding(
                root, "docs/t9_2_4_dual_backend_execution_manifest.json"
            ),
            "cell_ledger": _binding(
                root, "docs/t9_2_4_dual_backend_cell_ledger.csv"
            ),
            "raw_state_archive": _binding(
                root, "docs/t9_2_4_dual_backend_state_archive.npz"
            ),
            "source_data": _binding(
                root,
                "docs/t9_2_4_dual_backend_qualification_source_data.csv",
            ),
            "release_pin": _binding(
                root, "configs/phase9/t9_2_4_release_pin.json"
            ),
        },
        "summary": {
            "rows": snapshot["ledger_count"],
            "gate_passed": snapshot["source_passed_count"],
            "gate_failed": snapshot["source_failed_count"],
            "failure_counts": snapshot["failure_counts"],
            "worst_gate": snapshot["worst_gate"],
            "worst_point_estimate": snapshot["worst_independent_point"],
            "worst_simultaneous_bound": snapshot["worst_source_bound"],
            "worst_tolerance": snapshot["worst_tolerance"],
            "released_tasks": snapshot["released_tasks"],
            "blocked_tasks": snapshot["blocked_tasks"],
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "all_passed": all(gates.values()),
        },
        "targeted_mutations": mutations,
        "mutation_summary": {
            "detected": sum(item["detected"] is True for item in mutations),
            "total": len(mutations),
            "all_detected": all(
                item["detected"] is True for item in mutations
            ),
        },
        "claim_state": {key: None for key in EXPECTED_CLAIMS},
        "claim_boundary": (
            "Audit confirms evidence integrity and the frozen scientific "
            "NO-GO only; it does not qualify LER, lifetime, physical, "
            "hardware, Puviani, external-SOTA, or rank claims."
        ),
    }
    audit["analysis_sha256"] = _sha_bytes(
        _canonical(audit).encode("utf-8")
    )
    return audit


def write_audit(path: Path, audit: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/t9_2_4_dual_backend_post_outcome_audit.json"
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    audit = build_audit(root)
    write_audit(output, audit)
    print(
        _canonical(
            {
                "task_id": TASK_ID,
                "verdict": audit["verdict"],
                "analysis_sha256": audit["analysis_sha256"],
                "gates": audit["gate_summary"],
                "mutations": audit["mutation_summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_BLOCKED",
    "EXPECTED_CLAIMS",
    "EXPECTED_FAILURES",
    "EXPECTED_RELEASED",
    "SCHEMA",
    "TASK_ID",
    "VERDICT",
    "audit_snapshot",
    "build_audit",
    "build_snapshot",
    "main",
    "write_audit",
]
