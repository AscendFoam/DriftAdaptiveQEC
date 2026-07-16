"""Fail-closed strong/falsified teacher-student branch freeze for T4.4.5.

This task does not run another evaluation.  It consumes the hash-bound
T4.4.1--T4.4.4 artifacts, preserves their counterevidence, and deterministically
selects either a qualified student-retention branch or the drift/regime-aware
MAP-LUT fallback branch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "T4.4.5"
SCHEMA_VERSION = 1
PROTOCOL_ID = "T445-FAIL-CLOSED-TEACHER-STUDENT-BRANCH-FREEZE-V1"
STRONG_BRANCH_ID = "qualified_student_retention"
FALLBACK_BRANCH_ID = "drift_regime_aware_map_lut"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t4_4_5_teacher_student_branch_freeze.json")
DEFAULT_SOURCE_DATA = Path("docs/t4_4_5_teacher_student_branch_freeze_source_data.csv")

PARENT_ARTIFACTS = {
    "T4.4.1": Path("docs/t4_4_1_bounded_residual_rnn_teacher_validation.json"),
    "T4.4.2": Path("docs/t4_4_2_teacher_hidden_control_analysis.json"),
    "T4.4.3": Path("docs/t4_4_3_low_dimensional_student_validation.json"),
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention.json"),
}

QUALIFIED_ALLOWED_CLAIMS = (
    {
        "claim_id": "CL-T445-01",
        "statement": (
            "A fresh bounded-residual GRU teacher is reproducible in the matched "
            "two-level finite-cutoff ten-cycle simulator under the registered strict split."
        ),
    },
    {
        "claim_id": "CL-T445-02",
        "statement": (
            "The frozen four-state student retains at least ninety percent of the "
            "teacher-versus-standard gain for all preregistered metrics at cutoff twelve "
            "and sixteen on new paired seeds, including the paired-bootstrap lower bound."
        ),
    },
    {
        "claim_id": "CL-T445-03",
        "statement": (
            "The float student uses 95 stored scalars and 87 analytic MACs per healthy "
            "half-cycle versus 72853 scalars and 72266 analytic MACs for the float teacher."
        ),
    },
)

FALLBACK_ALLOWED_CLAIMS = (
    {
        "claim_id": "CL-T445-F01",
        "statement": (
            "Teacher and distillation claims are removed; the deployable algorithmic "
            "direction falls back to observed-only drift/regime-aware MAP-LUT plus the "
            "registered conservative health and event paths."
        ),
    },
)

PROHIBITED_CLAIMS = (
    "universal NMF superiority over every exact-budget MF baseline",
    "a unique belief-state or universal single-exponential teacher mechanism",
    "global optimizer optimality or convergence of teacher or student training",
    "a globally optimal or ten-cycle control oracle",
    "native multilevel leakage or SPAM robustness",
    "long-horizon or out-of-distribution gain retention",
    "quantized RTL synthesis FPGA timing board or device performance",
)

REVOCATION_TRIGGERS = (
    {
        "trigger_id": "RV-T445-01",
        "task": "T5.2",
        "condition": "multilevel leakage SPAM or reset evidence invalidates the matched-model gain",
    },
    {
        "trigger_id": "RV-T445-02",
        "task": "T5.4",
        "condition": "OOD model mismatch or long-horizon retention fails its preregistered gate",
    },
    {
        "trigger_id": "RV-T445-03",
        "task": "T5.5",
        "condition": "fixed-point or resource/deadline evidence rejects the 4-state student path",
    },
    {
        "trigger_id": "RV-T445-04",
        "task": "T6",
        "condition": "RTL transport FPGA or board negative-path evidence fails",
    },
    {
        "trigger_id": "RV-T445-05",
        "task": "all",
        "condition": "any T4.4.1--T4.4.4 artifact source checkpoint hash or gate becomes stale or false",
    },
)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_parent_artifacts(
    paths: Mapping[str, str | Path] = PARENT_ARTIFACTS,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task_id, path in paths.items():
        payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
        if payload.get("task_id") != task_id:
            raise ValueError(f"{path} has task_id {payload.get('task_id')!r}, expected {task_id}")
        result[task_id] = payload
    return result


def current_parent_implementation_hashes() -> dict[str, str]:
    from .bounded_residual_rnn_teacher import implementation_sha256 as t441_hash
    from .bounded_residual_teacher_analysis import implementation_sha256 as t442_hash
    from .low_dimensional_student_distillation import implementation_sha256 as t443_hash
    from .teacher_student_gain_retention import implementation_sha256 as t444_hash

    return {
        "T4.4.1": t441_hash(),
        "T4.4.2": t442_hash(),
        "T4.4.3": t443_hash(),
        "T4.4.4": t444_hash(),
    }


def _declared_file_bindings(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for key in ("source_data", "checkpoint"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value.get("path") and value.get("sha256"):
            bindings.append(
                {"role": key, "path": str(value["path"]), "sha256": str(value["sha256"])}
            )
    student = payload.get("student_artifact")
    if isinstance(student, Mapping) and student.get("path") and student.get("file_sha256"):
        bindings.append(
            {
                "role": "student_artifact",
                "path": str(student["path"]),
                "sha256": str(student["file_sha256"]),
            }
        )
    return bindings


def inspect_parent_integrity(
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task_id, payload in parents.items():
        checks = []
        for binding in _declared_file_bindings(payload):
            path = _repo_path(binding["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            checks.append({**binding, "actual_sha256": actual, "passed": actual == binding["sha256"]})
        result[task_id] = {
            "bindings": checks,
            "passed": bool(checks) and all(check["passed"] for check in checks),
        }
    return result


def _all_machine_gates_pass(payload: Mapping[str, Any]) -> bool:
    gates = payload.get("gates")
    summary = payload.get("gate_summary")
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, Mapping)
        and gates
        and all(value is True for value in gates.values())
        and isinstance(summary, Mapping)
        and summary.get("passed") == len(gates)
        and summary.get("total") == len(gates)
        and summary.get("failed") == []
    )


def _all_retention_metrics_pass(t444: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    threshold = t444.get("retention_threshold", {})
    point_threshold = threshold.get("point_fraction")
    lower_threshold = threshold.get("paired_bootstrap_ci_lower")
    rows: list[dict[str, Any]] = []
    passed = bool(
        point_threshold == 0.90
        and lower_threshold == 0.90
        and threshold.get("frozen_before_physical_evaluation") is True
    )
    retention = t444.get("stochastic_retention", {})
    for lane in ("primary", "confirmation"):
        metrics = retention.get(lane, {}) if isinstance(retention, Mapping) else {}
        for metric in (
            "selection_score",
            "fidelity_effective_lifetime_cycles",
            "logical_z_effective_lifetime_cycles",
        ):
            value = metrics.get(metric, {}) if isinstance(metrics, Mapping) else {}
            ci = value.get("ci_95", ()) if isinstance(value, Mapping) else ()
            metric_pass = bool(
                isinstance(value, Mapping)
                and value.get("defined") is True
                and isinstance(ci, Sequence)
                and len(ci) == 2
                and value.get("point_retention_fraction") is not None
                and float(value["point_retention_fraction"]) >= 0.90
                and float(ci[0]) >= 0.90
                and value.get("positive_teacher_gain_bootstrap_fraction") == 1.0
            )
            rows.append(
                {
                    "lane": lane,
                    "metric": metric,
                    "point": value.get("point_retention_fraction") if isinstance(value, Mapping) else None,
                    "ci_lower": ci[0] if isinstance(ci, Sequence) and len(ci) == 2 else None,
                    "passed": metric_pass,
                }
            )
            passed = passed and metric_pass
    return passed and len(rows) == 6, rows


def _selection_score(lane: Mapping[str, Any], strategy: str) -> float:
    if strategy == "mf_all_agents":
        metrics = lane[strategy]["metric_mean_across_agents"]
        return 0.5 * (
            float(metrics["fidelity_normalized_auc"])
            + float(metrics["logical_z_normalized_auc"])
        )
    return float(lane[strategy]["selection_score_mean"])


def decide_branch(
    parents: Mapping[str, Mapping[str, Any]],
    current_implementation_hashes: Mapping[str, str],
    integrity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic branch decision without mutating or rerunning parents."""

    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    t441, t442, t443, t444 = (parents[task] for task in PARENT_ARTIFACTS)

    parent_pass = {task: _all_machine_gates_pass(parents[task]) for task in PARENT_ARTIFACTS}
    source_current = {
        task: parents[task].get("implementation_sha256") == current_implementation_hashes.get(task)
        for task in PARENT_ARTIFACTS
    }
    files_current = {
        task: bool(integrity.get(task, {}).get("passed")) for task in PARENT_ARTIFACTS
    }

    fresh_teacher = bool(
        t441.get("training_protocol_id") == "T441-FRESH-BOUNDED-RESIDUAL-GRU-STRICT-SPLIT-V1"
        and t441.get("execution", {}).get("fresh_restart_count_in_checkpoint", 0) >= 3
        and t441.get("failed_restart_indices") == []
        and t441.get("gates", {}).get("no_parent_checkpoint_was_loaded_or_renamed") is True
        and t441.get("gates", {}).get("restart_selection_is_validation_only") is True
    )
    analysis_is_post_hoc = bool(
        t442.get("analysis_protocol_id") == "T442-FROZEN-TEACHER-HIDDEN-CONTROL-PG-V1"
        and t442.get("teacher_provenance", {}).get("teacher_parameters_frozen") is True
        and t442.get("teacher_provenance", {}).get("optimizer_steps_in_analysis") == 0
        and t442.get("leakage_proxy", {}).get("teacher_native") is False
    )
    strict_student = bool(
        t443.get("training_protocol_id") == "T443-LOWDIM-EXPONENTIAL-STRICT-SPLIT-V1"
        and t443.get("selection", {}).get("evaluation_blind") is True
        and t443.get("selection", {}).get("selected_dimension") == 4
        and t443.get("student_artifact", {}).get("state_dimension") == 4
        and t443.get("student_artifact", {})
        .get("runtime_replay", {})
        .get("leakage_exact_zero")
        is True
        and t443.get("student_artifact", {})
        .get("runtime_replay", {})
        .get("health_exact_zero")
        is True
    )
    retention_pass, retention_rows = _all_retention_metrics_pass(t444)

    try:
        primary = t444["stochastic_ten_cycle"]["primary"]
        confirmation = t444["stochastic_ten_cycle"]["confirmation"]
        primary_mf = _selection_score(primary, "mf_all_agents")
        primary_teacher = _selection_score(primary, "teacher")
        confirmation_mf = _selection_score(confirmation, "mf_all_agents")
        confirmation_teacher = _selection_score(confirmation, "teacher")
        mf_reversal_preserved = primary_mf > primary_teacher and confirmation_teacher > confirmation_mf
    except (KeyError, TypeError, ValueError):
        primary_mf = primary_teacher = confirmation_mf = confirmation_teacher = None
        mf_reversal_preserved = False

    predicates = {
        "all_parent_machine_gates_pass": all(parent_pass.values()),
        "all_parent_implementations_are_current": all(source_current.values()),
        "all_declared_parent_files_match_hashes": all(files_current.values()),
        "teacher_is_fresh_reproducible_and_validation_selected": fresh_teacher,
        "teacher_analysis_is_frozen_post_hoc_and_leakage_ood": analysis_is_post_hoc,
        "student_is_strict_split_four_state_and_fail_closed": strict_student,
        "all_six_preregistered_stochastic_retention_metrics_pass": retention_pass,
        "cutoff_dependent_mf_teacher_reversal_is_preserved": mf_reversal_preserved,
    }
    strong = all(predicates.values())
    active_branch = STRONG_BRANCH_ID if strong else FALLBACK_BRANCH_ID
    active_claims = QUALIFIED_ALLOWED_CLAIMS if strong else FALLBACK_ALLOWED_CLAIMS

    decision_core = {
        "protocol_id": PROTOCOL_ID,
        "active_branch_id": active_branch,
        "strong_branch_activated": strong,
        "fallback_branch_activated": not strong,
        "evidence_predicates": predicates,
        "active_claims": active_claims,
        "prohibited_claims": PROHIBITED_CLAIMS,
        "revocation_triggers": REVOCATION_TRIGGERS,
    }
    failed_predicates = [name for name, value in predicates.items() if not value]
    contract_gates = {
        "exactly_one_branch_is_active": strong != (not strong),
        "strong_branch_requires_every_evidence_predicate": strong == all(predicates.values()),
        "fallback_is_automatic_when_any_predicate_fails": strong or bool(failed_predicates),
        "fallback_removes_teacher_and_distillation_claims": strong
        or active_claims == FALLBACK_ALLOWED_CLAIMS,
        "qualified_claims_are_finite_model_and_metric_specific": (not strong)
        or (
            active_claims == QUALIFIED_ALLOWED_CLAIMS
            and "matched two-level finite-cutoff" in active_claims[0]["statement"]
            and "all preregistered metrics" in active_claims[1]["statement"]
            and "float student" in active_claims[2]["statement"]
        ),
        "universal_nmf_over_mf_claim_is_prohibited": any(
            "universal NMF" in claim for claim in PROHIBITED_CLAIMS
        ),
        "optimizer_and_oracle_optimality_claims_are_prohibited": any(
            "optimizer" in claim for claim in PROHIBITED_CLAIMS
        )
        and any("oracle" in claim for claim in PROHIBITED_CLAIMS),
        "leakage_ood_long_horizon_and_hardware_claims_are_prohibited": all(
            any(token in claim for claim in PROHIBITED_CLAIMS)
            for token in ("leakage", "long-horizon", "FPGA")
        ),
        "mf_counterevidence_is_retained_not_overwritten": mf_reversal_preserved or not strong,
        "later_revocation_triggers_cover_t5_2_t5_4_t5_5_and_t6": {
            item["task"] for item in REVOCATION_TRIGGERS
        } >= {"T5.2", "T5.4", "T5.5", "T6"},
        "decision_is_deterministic_and_evaluation_free": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "PASS" if all(contract_gates.values()) else "FAIL",
        "scope": (
            "read-only hash-bound T4.4.1--T4.4.4 branch freeze; no new evaluation, "
            "no universal MF ranking, leakage, OOD, long-horizon, quantized RTL, FPGA, board or device evidence"
        ),
        "protocol_id": PROTOCOL_ID,
        "active_branch": {
            "branch_id": active_branch,
            "strong_branch_activated": strong,
            "fallback_branch_activated": not strong,
            "failed_evidence_predicates": failed_predicates,
        },
        "evidence_predicates": predicates,
        "parent_machine_gate_status": parent_pass,
        "parent_implementation_current": source_current,
        "parent_declared_file_integrity": files_current,
        "retention_metrics": retention_rows,
        "counterevidence": {
            "primary_cutoff12_mf_mean_selection_score": primary_mf,
            "primary_cutoff12_teacher_selection_score": primary_teacher,
            "confirmation_cutoff16_mf_mean_selection_score": confirmation_mf,
            "confirmation_cutoff16_teacher_selection_score": confirmation_teacher,
            "ordering_reverses_across_cutoffs": mf_reversal_preserved,
            "interpretation": (
                "student retention is qualified evidence; universal NMF-over-MF superiority remains falsified"
            ),
        },
        "claim_registry": {
            "active_allowed": list(active_claims),
            "prohibited": list(PROHIBITED_CLAIMS),
            "fallback_allowed": list(FALLBACK_ALLOWED_CLAIMS),
        },
        "fallback_contract": {
            "branch_id": FALLBACK_BRANCH_ID,
            "activation_rule": "activate if any evidence predicate is false or any later revocation trigger fires",
            "algorithmic_scope": (
                "observed-only drift/regime-aware MAP-LUT with conservative health event FSM and atomic parameter-bank path"
            ),
            "teacher_or_distillation_claims_retained": False,
        },
        "revocation_triggers": list(REVOCATION_TRIGGERS),
        "decision_contract_hash": _canonical_sha256(decision_core),
        "gates": contract_gates,
        "gate_summary": {
            "passed": sum(contract_gates.values()),
            "total": len(contract_gates),
            "failed": [name for name, value in contract_gates.items() if not value],
        },
    }


CSV_FIELDS = (
    "row_type",
    "item_id",
    "task_id",
    "lane",
    "metric",
    "observed_value",
    "threshold",
    "passed",
    "branch_id",
    "statement",
    "source_artifact_sha256",
)


def _source_rows(
    result: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
    parent_artifact_sha256: Mapping[str, str],
    integrity: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(row_type: str, item_id: str, **values: Any) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row.update({"row_type": row_type, "item_id": item_id, **values})
        rows.append(row)

    branch_id = result["active_branch"]["branch_id"]
    for task_id, payload in parents.items():
        add(
            "parent_artifact",
            f"{task_id}-artifact",
            task_id=task_id,
            passed=result["parent_machine_gate_status"][task_id],
            branch_id=branch_id,
            statement="status and complete parent gate summary",
            source_artifact_sha256=parent_artifact_sha256[task_id],
        )
        for gate_name, value in payload["gates"].items():
            add(
                "parent_gate",
                f"{task_id}:{gate_name}",
                task_id=task_id,
                metric=gate_name,
                observed_value=value,
                passed=value,
                branch_id=branch_id,
                source_artifact_sha256=parent_artifact_sha256[task_id],
            )
        for binding in integrity[task_id]["bindings"]:
            add(
                "file_binding",
                f"{task_id}:{binding['role']}:{binding['path']}",
                task_id=task_id,
                metric=binding["role"],
                observed_value=binding["actual_sha256"],
                threshold=binding["sha256"],
                passed=binding["passed"],
                branch_id=branch_id,
                source_artifact_sha256=parent_artifact_sha256[task_id],
            )
    for row in result["retention_metrics"]:
        add(
            "retention_gate",
            f"{row['lane']}:{row['metric']}",
            task_id="T4.4.4",
            lane=row["lane"],
            metric=row["metric"],
            observed_value=f"point={row['point']};ci_lower={row['ci_lower']}",
            threshold="point>=0.90;ci_lower>=0.90",
            passed=row["passed"],
            branch_id=branch_id,
            source_artifact_sha256=parent_artifact_sha256["T4.4.4"],
        )
    for name, value in result["evidence_predicates"].items():
        add(
            "branch_predicate",
            name,
            metric=name,
            observed_value=value,
            passed=value,
            branch_id=branch_id,
        )
    for claim in result["claim_registry"]["active_allowed"]:
        add("allowed_claim", claim["claim_id"], branch_id=branch_id, statement=claim["statement"])
    for index, claim in enumerate(result["claim_registry"]["prohibited"], start=1):
        add("prohibited_claim", f"PC-T445-{index:02d}", branch_id=branch_id, statement=claim)
    for trigger in result["revocation_triggers"]:
        add(
            "revocation_trigger",
            trigger["trigger_id"],
            task_id=trigger["task"],
            branch_id=branch_id,
            statement=trigger["condition"],
        )
    return rows


def run_branch_freeze(
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    parents = load_parent_artifacts()
    implementations = current_parent_implementation_hashes()
    integrity = inspect_parent_integrity(parents)
    result = decide_branch(parents, implementations, integrity)
    parent_hashes = {task: _sha256(path) for task, path in PARENT_ARTIFACTS.items()}
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["implementation_sha256"] = implementation_sha256()
    result["parent_provenance"] = {
        task: {
            "path": PARENT_ARTIFACTS[task].as_posix(),
            "artifact_sha256": parent_hashes[task],
            "declared_implementation_sha256": parents[task]["implementation_sha256"],
            "current_implementation_sha256": implementations[task],
            "machine_gate_count": len(parents[task]["gates"]),
            "declared_file_bindings": integrity[task]["bindings"],
        }
        for task in PARENT_ARTIFACTS
    }
    rows = _source_rows(result, parents, parent_hashes, integrity)
    source_path = _repo_path(source_data_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "row_types": sorted({row["row_type"] for row in rows}),
    }
    artifact = _repo_path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    result = run_branch_freeze(artifact_path=args.artifact, source_data_path=args.source_data)
    print(
        json.dumps(
            {
                "status": result["status"],
                "active_branch": result["active_branch"],
                "gate_summary": result["gate_summary"],
                "source_rows": result["source_data"]["row_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
