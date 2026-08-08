"""T5.1.4 fail-closed algorithm-success/falsification branch freeze.

This module is intentionally read-only with respect to all parent evaluations.  It
binds the T5.1.1--T5.1.3 comparison evidence and the deployable MAP-LUT safety
contracts, then deterministically selects exactly one manuscript branch.  A PASS
means the branch decision is complete and provenance-current; it does not mean
that the strong learned-algorithm performance branch passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.1.4"
SCHEMA_VERSION = "t5.1.4-algorithm-success-falsification-v1"
PROTOCOL_ID = "FAIL-CLOSED-ALGORITHM-BRANCH-V1"
DEFAULT_ARTIFACT = Path("docs/t5_1_4_algorithm_branch_verdict.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_1_4_algorithm_branch_verdict_source_data.csv")

STRONG_BRANCH_ID = "matched_learned_decoder_performance"
FALLBACK_BRANCH_ID = "event_aware_adaptive_map_fpga_codesign"

PARENT_ARTIFACTS: dict[str, Path] = {
    "T5.1.1": Path("docs/t5_1_1_comparison_set_registry.json"),
    "T5.1.2": Path("docs/t5_1_2_mixed_scenario_matrix.json"),
    "T5.1.3": Path("docs/t5_1_3_oracle_gap_tail_report.json"),
    "T2.4.3": Path("docs/t2_4_3_fixed_point_validation.json"),
    "T4.2.1": Path("docs/t4_2_1_parametric_map_lut_validation.json"),
    "T4.2.3": Path("docs/t4_2_3_conservative_fallback_validation.json"),
    "T4.3.2": Path("docs/t4_3_2_atomic_parameter_bank_validation.json"),
    "T4.4.5": Path("docs/t4_4_5_teacher_student_branch_freeze.json"),
}

LEARNED_DECODER_TOKENS = ("cnn", "tcn", "gru", "rnn", "neural", "learned")
REQUIRED_REOPEN_GATES = (
    "new_independent_seed_clusters_preregistered_before_access",
    "learned_decoder_candidate_runs_same_trace_and_metric_lane",
    "static_window_ewma_kalman_and_applicable_sliding_baselines_matched",
    "static_average_p95_and_worst_non_degradation",
    "drift_or_regime_positive_seed_cluster_ci",
    "holm_adjusted_familywise_discovery",
    "no_preregistered_transient_tail_violation",
    "observed_only_causal_information_set",
    "deployment_scope_and_costs_reported",
    "t5_1_3_windows_not_relabelled_as_independent_seeds",
)

FALLBACK_ALLOWED_CLAIMS = (
    {
        "claim_id": "CL-T514-F01",
        "claim_type": "direction",
        "statement": (
            "The current paper direction is event/regime-aware, observed-only adaptive "
            "periodic MAP with an FPGA-oriented parameter-bank and fail-closed co-design; "
            "the registered event components are not yet a unified closed-loop performance result."
        ),
    },
    {
        "claim_id": "CL-T514-F02",
        "claim_type": "counterevidence",
        "statement": (
            "Classical adaptive MAP effect sizes are diagnostic only: the 24-comparison "
            "Holm family has zero discoveries and the calibration-shift Kalman transient "
            "worst window is worse than static."
        ),
    },
    {
        "claim_id": "CL-T514-F03",
        "claim_type": "software_contract",
        "statement": (
            "The fixed-point, parametric MAP-LUT, conservative fallback and atomic-bank "
            "evidence is software or hardware-aware validation only, not RTL, synthesis, "
            "board, device or measured hardware performance."
        ),
    },
)

STRONG_ALLOWED_CLAIMS = (
    {
        "claim_id": "CL-T514-S01",
        "claim_type": "conditional_performance",
        "statement": (
            "A named learned decoder candidate meets every preregistered static, drift, "
            "multiplicity, tail, causality and deployment gate on matched independent seeds."
        ),
    },
)

PROHIBITED_CLAIMS = (
    "CNN, TCN, GRU or another learned decoder outperforms static MAP or a strong adaptive baseline in the expanded T5.1 matrix",
    "a learned algorithm is superior under mixed drift or all registered noise classes",
    "adaptive decoding is universally superior to static MAP",
    "the four lane-local scenario models form one end-to-end closed-loop robustness experiment",
    "the qualified T4.4.5 controller teacher/student result is matched decoder evidence",
    "the historical T24 frozen-set result is expanded-matrix or independent-seed confirmation",
    "the finite-horizon control reference is a global or ten-cycle control oracle",
    "software fixed-point or LUT evidence is measured FPGA, board or device performance",
)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_parent_artifacts(
    paths: Mapping[str, str | Path] = PARENT_ARTIFACTS,
) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for task_id, path in paths.items():
        payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
        if payload.get("task_id") != task_id:
            raise ValueError(
                f"{path} has task_id {payload.get('task_id')!r}, expected {task_id}"
            )
        parents[task_id] = payload
    return parents


def _gate_entries(payload: Mapping[str, Any]) -> list[tuple[str, bool]]:
    gates = payload.get("gates")
    if isinstance(gates, Mapping):
        return [(str(name), value is True) for name, value in gates.items()]
    if isinstance(gates, list):
        return [
            (str(item.get("id", index)), item.get("passed") is True)
            for index, item in enumerate(gates)
            if isinstance(item, Mapping)
        ]
    summary = payload.get("gate_summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("gates"), Mapping):
        return [
            (str(name), value is True) for name, value in summary["gates"].items()
        ]
    return []


def _machine_pass(payload: Mapping[str, Any]) -> bool:
    gates = _gate_entries(payload)
    return bool(payload.get("status") == "PASS" and gates and all(v for _, v in gates))


def _declared_file_bindings(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []

    def add(role: str, record: Any, *, path_key: str = "path", sha_key: str = "sha256") -> None:
        if isinstance(record, Mapping) and record.get(path_key) and record.get(sha_key):
            bindings.append(
                {
                    "role": role,
                    "path": str(record[path_key]),
                    "declared_sha256": str(record[sha_key]),
                }
            )

    add("source_data", payload.get("source_data"))
    add("image_artifact", payload.get("image_artifact"))
    for index, record in enumerate(payload.get("artifact_bindings", ())):
        add(f"artifact_binding:{index}", record)
    for index, record in enumerate(payload.get("implementation_bindings", ())):
        add(f"implementation_binding:{index}", record)
    provenance = payload.get("parent_provenance")
    if isinstance(provenance, Mapping):
        for task_id, record in provenance.items():
            add(
                f"parent_provenance:{task_id}",
                record,
                sha_key="artifact_sha256",
            )
            if isinstance(record, Mapping):
                for index, child in enumerate(record.get("declared_file_bindings", ())):
                    add(f"parent_file:{task_id}:{index}", child)
    return bindings


def _fixed_point_composite_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/fixed_point_chain.py",
        "cnn_fpga/decoder/linear_runtime.py",
        "physics/ideal_gkp_decoder.py",
    ):
        encoded = relative.encode("utf-8")
        content = _repo_path(relative).read_bytes()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def current_parent_composite_hashes() -> dict[str, str]:
    from cnn_fpga.benchmark.atomic_parameter_bank_validation import (
        _implementation_sha256 as atomic_hash,
    )
    from cnn_fpga.benchmark.conservative_fallback_validation import (
        _implementation_sha256 as fallback_hash,
    )
    from cnn_fpga.benchmark.parametric_map_lut_validation import (
        _implementation_sha256 as lut_hash,
    )
    from cnn_fpga.benchmark.teacher_student_branch_freeze import (
        implementation_sha256 as teacher_student_hash,
    )

    return {
        "T2.4.3": _fixed_point_composite_sha256(),
        "T4.2.1": lut_hash(),
        "T4.2.3": fallback_hash(),
        "T4.3.2": atomic_hash(),
        "T4.4.5": teacher_student_hash(),
    }


def inspect_parent_integrity(
    parents: Mapping[str, Mapping[str, Any]],
    composite_hashes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    current_composites = dict(
        current_parent_composite_hashes() if composite_hashes is None else composite_hashes
    )
    result: dict[str, dict[str, Any]] = {}
    for task_id, payload in parents.items():
        checks: list[dict[str, Any]] = []
        for binding in _declared_file_bindings(payload):
            path = _repo_path(binding["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            checks.append(
                {
                    **binding,
                    "actual_sha256": actual,
                    "passed": actual == binding["declared_sha256"],
                }
            )
        declared_composite = payload.get("implementation_sha256")
        composite_current = True
        if task_id in current_composites:
            composite_current = declared_composite == current_composites[task_id]
        result[task_id] = {
            "machine_pass": _machine_pass(payload),
            "machine_gate_count": len(_gate_entries(payload)),
            "declared_file_bindings": checks,
            "all_declared_files_current": all(item["passed"] for item in checks),
            "declared_composite_sha256": declared_composite,
            "current_composite_sha256": current_composites.get(task_id),
            "composite_current": composite_current,
        }
        result[task_id]["passed"] = bool(
            result[task_id]["machine_pass"]
            and result[task_id]["all_declared_files_current"]
            and composite_current
        )
    return result


def _scenario(report: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    rows = report.get("decoder_lane", {}).get("scenario_reports", ())
    matches = [row for row in rows if row.get("scenario_id") == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {scenario_id!r} scenario report, found {len(matches)}")
    return matches[0]


def _extract_evidence(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    registry = parents["T5.1.1"]
    matrix = parents["T5.1.2"]
    report = parents["T5.1.3"]
    teacher_student = parents["T4.4.5"]

    executed = tuple(matrix["decoder_lane"]["executed_comparators"])
    matched_learned = tuple(
        name for name in executed if any(token in name.lower() for token in LEARNED_DECODER_TOKENS)
    )
    learned_registered_wrong_lane = tuple(
        row["comparator_id"]
        for row in registry["comparators"]
        if any(token in row["comparator_id"].lower() for token in LEARNED_DECODER_TOKENS)
        and "decoder_continuous_drift" not in row.get("eligible_lanes", ())
    )
    multiplicity = report["decoder_lane"]["multiplicity"]
    static = _scenario(report, "static_gaussian")["methods"]
    shift = _scenario(report, "calibration_shift")["methods"]
    static_method = static["static"]
    static_kalman = static["kalman"]
    shift_static = shift["static"]
    shift_kalman = shift["kalman"]

    teacher_scope = str(teacher_student.get("scope", ""))
    teacher_claims = " ".join(
        str(row.get("statement", ""))
        for row in teacher_student.get("claim_registry", {}).get("active_allowed", ())
    )
    return {
        "decoder_lane": {
            "executed_comparators": list(executed),
            "matched_learned_decoder_candidates": list(matched_learned),
            "registered_learned_or_neural_rows_outside_decoder_lane": list(
                learned_registered_wrong_lane
            ),
            "same_trace_contract": matrix["decoder_lane"]["shared_trace_contract"],
            "independent_unit": report["decoder_lane"]["independent_unit"],
            "seed_cluster_count": len(report["decoder_lane"]["config"]["evaluation_seeds"])
            * len(report["decoder_lane"]["scenario_reports"]),
        },
        "multiplicity": {
            "hypotheses": multiplicity["hypotheses"],
            "adjustment": multiplicity["adjustment"],
            "discoveries": multiplicity["discoveries"],
            "minimum_raw_p_value": min(row["raw_p_value"] for row in multiplicity["rows"]),
            "minimum_adjusted_p_value": min(
                row["holm_adjusted_p_value"] for row in multiplicity["rows"]
            ),
        },
        "classical_diagnostic": {
            "static_gaussian": {
                "static_p_l": static_method["p_l"],
                "kalman_p_l": static_kalman["p_l"],
                "static_window_ler_p95": static_method["window_ler_p95"],
                "kalman_window_ler_p95": static_kalman["window_ler_p95"],
                "static_observed_worst_window_ler": static_method[
                    "observed_worst_window_ler"
                ],
                "kalman_observed_worst_window_ler": static_kalman[
                    "observed_worst_window_ler"
                ],
            },
            "calibration_shift": {
                "static_p_l": shift_static["p_l"],
                "kalman_p_l": shift_kalman["p_l"],
                "static_window_ler_p95": shift_static["window_ler_p95"],
                "kalman_window_ler_p95": shift_kalman["window_ler_p95"],
                "static_observed_worst_window_ler": shift_static[
                    "observed_worst_window_ler"
                ],
                "kalman_observed_worst_window_ler": shift_kalman[
                    "observed_worst_window_ler"
                ],
                "transient_tail_violation": shift_kalman[
                    "observed_worst_window_ler"
                ]
                > shift_static["observed_worst_window_ler"],
            },
            "interpretation": (
                "classical adaptive MAP motivates the fallback direction only; it is not "
                "a learned-candidate result and does not override multiplicity or tail evidence"
            ),
        },
        "teacher_student_separation": {
            "artifact_task_id": teacher_student.get("task_id"),
            "scope": teacher_scope,
            "active_branch_id": teacher_student.get("active_branch", {}).get("branch_id"),
            "controller_matched_model_only": bool(
                "T4.4.1--T4.4.4" in teacher_scope
                and "ten-cycle simulator" in teacher_claims
                and "decoder" not in teacher_claims.lower()
            ),
            "usable_as_t5_1_decoder_evidence": False,
        },
    }


def validate_branch_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return semantic errors; used by tests to reject narrative verdict edits."""

    errors: list[str] = []
    evidence = payload.get("evidence_snapshot", {})
    predicates = payload.get("strong_branch_predicates", {})
    active = payload.get("active_branch", {})
    claims = payload.get("claim_registry", {})
    fallback = payload.get("fallback_contract", {})
    reopen = payload.get("reopen_contract", {})

    expected_strong = bool(predicates) and all(value is True for value in predicates.values())
    expected_id = STRONG_BRANCH_ID if expected_strong else FALLBACK_BRANCH_ID
    if active.get("branch_id") != expected_id:
        errors.append("active branch does not equal conjunction of strong predicates")
    if active.get("strong_branch_activated") is not expected_strong:
        errors.append("strong branch flag is inconsistent")
    if active.get("fallback_branch_activated") is not (not expected_strong):
        errors.append("fallback branch flag is inconsistent")

    decoder = evidence.get("decoder_lane", {})
    matched = decoder.get("matched_learned_decoder_candidates", ())
    if predicates.get("matched_learned_decoder_candidate_executed") is not bool(matched):
        errors.append("matched learned-candidate predicate is inconsistent")
    multiplicity = evidence.get("multiplicity", {})
    if multiplicity.get("discoveries") != 0:
        errors.append("committed T5.1.3 zero-discovery counterevidence was changed")
    if predicates.get("holm_adjusted_repeatable_advantage") is not False:
        errors.append("Holm predicate must remain false for zero discoveries")
    shift = evidence.get("classical_diagnostic", {}).get("calibration_shift", {})
    transient = bool(
        shift.get("kalman_observed_worst_window_ler", -1)
        > shift.get("static_observed_worst_window_ler", float("inf"))
    )
    if shift.get("transient_tail_violation") is not transient or not transient:
        errors.append("calibration-shift transient tail counterevidence was removed")
    if predicates.get("no_preregistered_transient_tail_violation") is not False:
        errors.append("tail predicate must remain false")

    separation = evidence.get("teacher_student_separation", {})
    if separation.get("controller_matched_model_only") is not True:
        errors.append("T4.4.5 matched-controller scope is not preserved")
    if separation.get("usable_as_t5_1_decoder_evidence") is not False:
        errors.append("T4.4.5 was incorrectly promoted to decoder evidence")

    active_claims = claims.get("active_allowed", ())
    if not expected_strong:
        if tuple(active_claims) != FALLBACK_ALLOWED_CLAIMS:
            errors.append("fallback active claim set changed")
        if fallback.get("cnn_or_learned_performance_claim_retained") is not False:
            errors.append("fallback retains a learned performance claim")
    if tuple(claims.get("prohibited", ())) != PROHIBITED_CLAIMS:
        errors.append("prohibited claim registry changed")

    prereqs = fallback.get("prerequisites", {})
    if not prereqs or not all(value is True for value in prereqs.values()):
        errors.append("fallback deployability prerequisites are incomplete")
    if fallback.get("hardware_measurement_claimed") is not False:
        errors.append("software evidence was promoted to measured hardware")

    gates = tuple(reopen.get("required_gates", ()))
    if set(gates) != set(REQUIRED_REOPEN_GATES) or len(gates) != len(REQUIRED_REOPEN_GATES):
        errors.append("reopen contract is incomplete")
    if reopen.get("existing_1152_windows_may_count_as_independent_seeds") is not False:
        errors.append("existing windows were allowed to masquerade as independent seeds")
    if reopen.get("new_seed_registration_timing") != "before_any_new_evaluation_access":
        errors.append("new seeds are not preregistered before access")
    return tuple(errors)


def decide_branch(
    parents: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    missing_integrity = set(PARENT_ARTIFACTS) - set(integrity)
    if missing_integrity:
        raise ValueError(f"missing parent integrity: {sorted(missing_integrity)}")

    evidence = _extract_evidence(parents)
    source_current = all(bool(integrity[task]["passed"]) for task in PARENT_ARTIFACTS)
    matched = bool(evidence["decoder_lane"]["matched_learned_decoder_candidates"])
    discoveries = int(evidence["multiplicity"]["discoveries"])
    transient = bool(
        evidence["classical_diagnostic"]["calibration_shift"]["transient_tail_violation"]
    )

    strong_predicates = {
        "all_parent_evidence_is_passed_and_current": source_current,
        "matched_learned_decoder_candidate_executed": matched,
        "candidate_compared_on_same_trace_to_static_and_strong_adaptive_baselines": matched,
        "static_average_p95_and_worst_non_degradation": matched,
        "positive_seed_cluster_effect_against_strong_deployable_baseline": matched
        and discoveries > 0,
        "holm_adjusted_repeatable_advantage": discoveries > 0,
        "no_preregistered_transient_tail_violation": matched and not transient,
        "candidate_is_observed_only_causal_and_deployment_scoped": matched,
    }
    strong = all(strong_predicates.values())
    active_claims = STRONG_ALLOWED_CLAIMS if strong else FALLBACK_ALLOWED_CLAIMS

    fallback_prerequisites = {
        "observed_only_adaptive_map_rows_are_registered": {
            "ewma_adaptive_map",
            "kalman_adaptive_map",
            "sliding_window_map",
        }.issubset(
            {row["comparator_id"] for row in parents["T5.1.1"]["comparators"]}
        ),
        "adaptive_map_methods_were_executed_in_decoder_lane": {
            "window",
            "ewma",
            "kalman",
        }.issubset(set(evidence["decoder_lane"]["executed_comparators"])),
        "event_and_regime_components_are_registered_but_component_only": all(
            any(
                row["comparator_id"] == comparator
                and row.get("ranking_status") == "component_only"
                for row in parents["T5.1.1"]["comparators"]
            )
            for comparator in ("run_length_event_controller", "regime_hmm_estimator")
        ),
        "fixed_point_contract_passes": bool(integrity["T2.4.3"]["passed"]),
        "parametric_map_lut_contract_passes": bool(integrity["T4.2.1"]["passed"]),
        "conservative_fallback_contract_passes": bool(integrity["T4.2.3"]["passed"]),
        "atomic_parameter_bank_contract_passes": bool(integrity["T4.3.2"]["passed"]),
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "the fail-closed branch decision, counterevidence retention, provenance and "
            "claim routing pass; PASS does not mean the strong learned-algorithm branch passed"
        ),
        "active_branch": {
            "branch_id": STRONG_BRANCH_ID if strong else FALLBACK_BRANCH_ID,
            "strong_branch_activated": strong,
            "fallback_branch_activated": not strong,
            "failed_strong_predicates": [
                name for name, value in strong_predicates.items() if not value
            ],
        },
        "strong_branch_predicates": strong_predicates,
        "evidence_snapshot": evidence,
        "parent_integrity": dict(integrity),
        "claim_registry": {
            "active_allowed": list(active_claims),
            "prohibited": list(PROHIBITED_CLAIMS),
            "historical_quarantine": {
                "T24_PC01": (
                    "historical frozen-set scope only; not expanded T5.1 evidence and not "
                    "independent confirmation"
                ),
                "T4.4.5": (
                    "qualified matched-controller teacher/student evidence only; not a "
                    "CNN or decoder performance result"
                ),
            },
        },
        "fallback_contract": {
            "branch_id": FALLBACK_BRANCH_ID,
            "activation_rule": "automatic if any strong predicate is false",
            "paper_direction": (
                "event/regime-aware observed-only adaptive periodic MAP with FPGA-oriented "
                "parameter-bank, conservative fallback and atomic update co-design"
            ),
            "current_integration_status": (
                "registered components and lane-local evidence; not yet one finite-energy "
                "end-to-end closed-loop performance matrix"
            ),
            "prerequisites": fallback_prerequisites,
            "cnn_or_learned_performance_claim_retained": False,
            "hardware_measurement_claimed": False,
        },
        "reopen_contract": {
            "required_gates": list(REQUIRED_REOPEN_GATES),
            "new_seed_registration_timing": "before_any_new_evaluation_access",
            "existing_1152_windows_may_count_as_independent_seeds": False,
            "reopen_authority": "a future explicit task-board item after preregistration",
        },
        "determinism_contract": {
            "parent_evaluations_rerun": False,
            "new_random_samples_generated": False,
            "decision_rule": "logical_conjunction_of_frozen_strong_predicates",
        },
    }

    semantic_errors = validate_branch_payload(result)
    contract_gates = {
        "all_parent_evidence_is_passed_and_current": source_current,
        "exactly_one_branch_is_active": strong != (not strong),
        "strong_branch_equals_all_predicates": strong == all(strong_predicates.values()),
        "fallback_automatically_activates_on_failure": strong
        or bool(result["active_branch"]["failed_strong_predicates"]),
        "matched_learned_candidate_absence_is_retained": not matched,
        "zero_holm_discoveries_are_retained": discoveries == 0,
        "calibration_shift_transient_counterevidence_is_retained": transient,
        "cnn_performance_claim_is_removed_in_fallback": strong
        or result["fallback_contract"]["cnn_or_learned_performance_claim_retained"]
        is False,
        "teacher_student_controller_evidence_is_not_decoder_evidence": evidence[
            "teacher_student_separation"
        ]["controller_matched_model_only"]
        and not evidence["teacher_student_separation"]["usable_as_t5_1_decoder_evidence"],
        "historical_t24_claim_is_quarantined": "historical frozen-set" in result[
            "claim_registry"
        ]["historical_quarantine"]["T24_PC01"],
        "fallback_prerequisites_pass": all(fallback_prerequisites.values()),
        "software_evidence_is_not_hardware_measurement": not result["fallback_contract"][
            "hardware_measurement_claimed"
        ],
        "independent_unit_remains_evaluation_seed": evidence["decoder_lane"][
            "independent_unit"
        ]
        == "evaluation_seed",
        "reopen_contract_is_complete_and_forbids_window_pseudoreplication": set(
            result["reopen_contract"]["required_gates"]
        )
        == set(REQUIRED_REOPEN_GATES)
        and not result["reopen_contract"][
            "existing_1152_windows_may_count_as_independent_seeds"
        ],
        "semantic_validator_accepts_only_the_derived_verdict": semantic_errors == (),
        "decision_is_read_only_and_deterministic": not result["determinism_contract"][
            "parent_evaluations_rerun"
        ]
        and not result["determinism_contract"]["new_random_samples_generated"],
    }
    result["gates"] = contract_gates
    result["gate_summary"] = {
        "passed": sum(contract_gates.values()),
        "total": len(contract_gates),
        "failed": [name for name, value in contract_gates.items() if not value],
    }
    result["status"] = "PASS" if all(contract_gates.values()) else "FAIL"
    decision_core = {
        "protocol_id": PROTOCOL_ID,
        "active_branch": result["active_branch"],
        "strong_branch_predicates": strong_predicates,
        "evidence_snapshot": evidence,
        "claim_registry": result["claim_registry"],
        "fallback_contract": result["fallback_contract"],
        "reopen_contract": result["reopen_contract"],
    }
    result["decision_contract_sha256"] = _canonical_sha256(decision_core)
    return result


CSV_FIELDS = (
    "row_type",
    "item_id",
    "task_id",
    "metric",
    "observed_value",
    "threshold_or_role",
    "passed",
    "branch_id",
    "statement",
    "source_artifact_sha256",
)


def _source_rows(
    result: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(row_type: str, item_id: str, **values: Any) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row.update({"row_type": row_type, "item_id": item_id, **values})
        rows.append(row)

    branch_id = result["active_branch"]["branch_id"]
    for task_id, payload in parents.items():
        integrity = result["parent_integrity"][task_id]
        add(
            "parent_artifact",
            task_id,
            task_id=task_id,
            observed_value=integrity["passed"],
            passed=integrity["passed"],
            branch_id=branch_id,
            statement="machine gates, file bindings and composite implementation current",
            source_artifact_sha256=artifact_hashes[task_id],
        )
        for gate_name, value in _gate_entries(payload):
            add(
                "parent_gate",
                f"{task_id}:{gate_name}",
                task_id=task_id,
                metric=gate_name,
                observed_value=value,
                passed=value,
                branch_id=branch_id,
                source_artifact_sha256=artifact_hashes[task_id],
            )
        for index, binding in enumerate(integrity["declared_file_bindings"]):
            add(
                "file_binding",
                f"{task_id}:{index}:{binding['role']}",
                task_id=task_id,
                metric=binding["path"],
                observed_value=binding["actual_sha256"],
                threshold_or_role=binding["declared_sha256"],
                passed=binding["passed"],
                branch_id=branch_id,
                source_artifact_sha256=artifact_hashes[task_id],
            )
    for name, value in result["strong_branch_predicates"].items():
        add(
            "strong_predicate",
            name,
            metric=name,
            observed_value=value,
            threshold_or_role="all must be true for strong branch",
            passed=value,
            branch_id=branch_id,
        )
    for name, value in result["fallback_contract"]["prerequisites"].items():
        add(
            "fallback_prerequisite",
            name,
            metric=name,
            observed_value=value,
            passed=value,
            branch_id=branch_id,
        )
    multi = result["evidence_snapshot"]["multiplicity"]
    for metric in (
        "hypotheses",
        "discoveries",
        "minimum_raw_p_value",
        "minimum_adjusted_p_value",
    ):
        add(
            "counterevidence",
            f"multiplicity:{metric}",
            task_id="T5.1.3",
            metric=metric,
            observed_value=multi[metric],
            branch_id=branch_id,
            source_artifact_sha256=artifact_hashes["T5.1.3"],
        )
    for scenario_id, metrics in result["evidence_snapshot"]["classical_diagnostic"].items():
        if not isinstance(metrics, Mapping):
            continue
        for metric, value in metrics.items():
            add(
                "counterevidence",
                f"{scenario_id}:{metric}",
                task_id="T5.1.3",
                metric=metric,
                observed_value=value,
                branch_id=branch_id,
                source_artifact_sha256=artifact_hashes["T5.1.3"],
            )
    for claim in result["claim_registry"]["active_allowed"]:
        add(
            "active_claim",
            claim["claim_id"],
            threshold_or_role=claim["claim_type"],
            branch_id=branch_id,
            statement=claim["statement"],
        )
    for index, statement in enumerate(result["claim_registry"]["prohibited"], start=1):
        add(
            "prohibited_claim",
            f"PC-T514-{index:02d}",
            branch_id=branch_id,
            statement=statement,
        )
    for gate in result["reopen_contract"]["required_gates"]:
        add(
            "reopen_gate",
            gate,
            threshold_or_role="required before strong branch may be reconsidered",
            branch_id=branch_id,
        )
    for name, value in result["gates"].items():
        add(
            "contract_gate",
            name,
            metric=name,
            observed_value=value,
            passed=value,
            branch_id=branch_id,
        )
    return rows


def write_artifacts(
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    parents = load_parent_artifacts()
    integrity = inspect_parent_integrity(parents)
    result = decide_branch(parents, integrity)
    artifact_hashes = {
        task_id: _sha256(path) for task_id, path in PARENT_ARTIFACTS.items()
    }
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["implementation_sha256"] = implementation_sha256()
    result["artifact_bindings"] = [
        {
            "task_id": task_id,
            "path": path.as_posix(),
            "sha256": artifact_hashes[task_id],
            "machine_and_integrity_pass": integrity[task_id]["passed"],
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    ]
    rows = _source_rows(result, parents, artifact_hashes)
    source_path = _repo_path(source_data_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
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
    artifact.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    result = write_artifacts(
        artifact_path=args.artifact, source_data_path=args.source_data
    )
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


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "FALLBACK_BRANCH_ID",
    "PARENT_ARTIFACTS",
    "REQUIRED_REOPEN_GATES",
    "STRONG_BRANCH_ID",
    "current_parent_composite_hashes",
    "decide_branch",
    "implementation_sha256",
    "inspect_parent_integrity",
    "load_parent_artifacts",
    "validate_branch_payload",
    "write_artifacts",
]
