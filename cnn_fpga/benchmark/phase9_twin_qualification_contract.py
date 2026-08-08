"""Pre-formal T9.2.4 dual-backend qualification seal.

This module intentionally does not import or execute either physics backend.
It freezes parent hashes, cells, seeds, tolerances, failure policy and claim
boundaries before the formal comparison runner may access an outcome.
"""

from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping


TASK_ID = "T9.2.4"
SCHEMA = "PHASE9-DUAL-BACKEND-QUALIFICATION-PREREGISTRATION-V1"
PASS_VERDICT = "PASS_T9_2_4_PREREGISTRATION_FROZEN"
EXPECTED_TOP_LEVEL = {
    "task_id",
    "schema_version",
    "frozen_at",
    "formal_result_accessed_before_freeze",
    "parent_a",
    "parent_b",
    "action_contract",
    "common_physics",
    "native_logical_mappings",
    "splits",
    "formal_matrix",
    "prefrozen_tolerances",
    "failure_policy",
    "artifact_paths",
    "claim_boundary",
}
EXPECTED_ARTIFACTS = {
    "preregistration_seal",
    "report",
    "cell_ledger",
    "source_data",
    "markdown",
    "release_pin",
}
EXPECTED_PROBES = tuple(f"P{index:02d}_{suffix}" for index, suffix in (
    (1, "IDLE"),
    (2, "Q_POS"),
    (3, "Q_NEG"),
    (4, "P_POS"),
    (5, "P_NEG"),
    (6, "ALTERNATE"),
    (7, "BOUNDARY"),
    (8, "PHASE"),
    (9, "LEAK_RESET"),
    (10, "RESET_OK"),
    (11, "RESET_FAIL"),
    (12, "BAD_CRC"),
    (13, "STALE"),
    (14, "OOD"),
    (15, "DEADLINE"),
    (16, "LKG_RECOVERY"),
))


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


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _binding(root: Path, relative: str) -> dict[str, object]:
    normalized = relative.replace("\\", "/")
    payload = (root / normalized).read_bytes()
    return {
        "path": normalized,
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold an object")
    return value


def _strict_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not 0.0 < number < float("inf"):
        raise ValueError(f"{name} must be finite and positive")
    return number


def _interval(split: Mapping[str, Any], name: str) -> tuple[int, int]:
    if set(split) != {"start", "count"}:
        raise ValueError(f"{name} seed schema mismatch")
    start = split["start"]
    count = split["count"]
    if (
        isinstance(start, bool)
        or isinstance(count, bool)
        or not isinstance(start, int)
        or not isinstance(count, int)
        or start < 0
        or count <= 0
    ):
        raise ValueError(f"{name} seed interval invalid")
    return start, start + count


def _released_parent(
    root: Path,
    spec: Mapping[str, Any],
    expected_task: str,
) -> dict[str, Any]:
    if spec.get("task_id") != expected_task:
        raise ValueError("parent task mismatch")
    pin = _load(root / str(spec["release_pin"]))
    report = _load(root / str(spec["report"]))
    analysis = str(spec["analysis_sha256"])
    if (
        pin.get("task_id") != expected_task
        or report.get("task_id") != expected_task
        or pin.get("analysis_sha256") != analysis
        or report.get("analysis_sha256") != analysis
    ):
        raise ValueError(f"{expected_task} analysis binding mismatch")
    if pin.get("report") != _binding(root, str(spec["report"])):
        raise ValueError(f"{expected_task} report hash drift")
    if expected_task == "T9.2.1":
        values = [
            value
            for group in report["current_null_state"].values()
            for value in group.values()
        ]
    else:
        values = list(report["qualification"]["claim_state"].values())
    if not values or any(value is not None for value in values):
        raise ValueError(f"{expected_task} has non-null outcome claims")
    return {
        "task_id": expected_task,
        "analysis_sha256": analysis,
        "release_pin": _binding(root, str(spec["release_pin"])),
        "report": _binding(root, str(spec["report"])),
        "all_outcome_claims_null": True,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != EXPECTED_TOP_LEVEL:
        raise ValueError("T9.2.4 config schema mismatch")
    if config["task_id"] != TASK_ID:
        raise ValueError("task id mismatch")
    if (
        config["schema_version"]
        != "PHASE9-DUAL-BACKEND-QUALIFICATION-CONFIG-V1"
    ):
        raise ValueError("config schema version mismatch")
    if config["formal_result_accessed_before_freeze"] is not False:
        raise ValueError("formal results were accessed before freeze")
    if set(config["artifact_paths"]) != EXPECTED_ARTIFACTS:
        raise ValueError("artifact path schema mismatch")
    if tuple(config["action_contract"]["probe_ids"]) != EXPECTED_PROBES:
        raise ValueError("representative probe ledger mismatch")
    matrix = config["formal_matrix"]
    if matrix["logical_labels"] != ["0", "1", "+", "-", "+i", "-i"]:
        raise ValueError("six-state ledger mismatch")
    if matrix["unique_nominal_actions"] != [
        "IDLE",
        "X",
        "Z",
        "XZ",
        "RESET",
        "HOLD",
        "LKG_HOLD",
    ]:
        raise ValueError("nominal action ledger mismatch")
    if set(matrix["fault_scenarios"]) != {
        "step",
        "telegraph",
        "burst",
        "compound",
    }:
        raise ValueError("fault scenario ledger mismatch")
    if matrix["all_representative_probes_required"] is not True:
        raise ValueError("all probes must remain required")
    for key in (
        "samples_per_shared_state_action_backend",
        "samples_per_representative_probe_backend",
        "samples_per_logical_state_action_backend",
        "trajectories_per_fault_backend",
        "bootstrap_resamples",
    ):
        value = matrix[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 16:
            raise ValueError(f"{key} is demo-sized or invalid")
    confidence = _strict_positive_number(matrix["confidence"], "confidence")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5,1)")
    tolerances = config["prefrozen_tolerances"]
    if set(tolerances) != {
        "minimum_code_principal_singular_value",
        "maximum_code_projector_frobenius",
        "maximum_ensemble_density_trace_distance",
        "maximum_mean_photon_difference",
        "maximum_level_probability_l1",
        "maximum_integrated_iq_mean_difference",
        "maximum_integrated_iq_covariance_frobenius",
        "maximum_iq_two_sample_ks",
        "maximum_log_evidence_mean_difference",
        "maximum_posterior_mean_l1",
        "maximum_logical_ptm_entry_difference",
        "maximum_logical_survival_difference",
        "maximum_reset_success_rate_difference",
        "maximum_leakage_residence_rate_difference",
        "maximum_short_trajectory_terminal_trace_distance",
        "maximum_short_trajectory_observable_mean_difference",
        "minimum_conservation_pass_fraction",
        "familywise_alpha",
    }:
        raise ValueError("tolerance schema mismatch")
    for key, value in tolerances.items():
        number = _strict_positive_number(value, key)
        if number > 1.0 and key != "maximum_code_projector_frobenius":
            raise ValueError(f"{key} exceeds a defensible normalized bound")
    failure = config["failure_policy"]
    for key in (
        "no_retuning_after_formal_access",
        "no_mean_only_rescue",
        "no_cell_deletion",
        "no_codebook_claim",
    ):
        if failure[key] is not True:
            raise ValueError(f"{key} must remain true")
    if failure["fail_verdict"] != "NO_GO_TWIN_QUALIFICATION":
        raise ValueError("fail verdict mismatch")
    if config["claim_boundary"]["typed_null"] != [
        "round_ler",
        "six_state_lifetime",
        "physical_break_even",
        "official_puviani_exact",
        "puviani_nmf_surpass",
        "external_sota",
        "hardware_measured",
        "rank",
    ]:
        raise ValueError("typed-null boundary mismatch")


def _cell_accounting(config: Mapping[str, Any]) -> dict[str, int]:
    matrix = config["formal_matrix"]
    backends = 2
    shared = (
        len(matrix["shared_fock_states"])
        * len(matrix["unique_nominal_actions"])
        * matrix["samples_per_shared_state_action_backend"]
        * backends
    )
    probes = (
        len(config["action_contract"]["probe_ids"])
        * matrix["samples_per_representative_probe_backend"]
        * backends
    )
    logical = (
        len(matrix["logical_labels"])
        * len(matrix["unique_nominal_actions"])
        * matrix["samples_per_logical_state_action_backend"]
        * backends
    )
    trajectories = (
        len(matrix["fault_scenarios"])
        * matrix["trajectories_per_fault_backend"]
        * backends
    )
    trajectory_rounds = sum(
        scenario["horizon"]
        * matrix["trajectories_per_fault_backend"]
        * backends
        for scenario in matrix["fault_scenarios"].values()
    )
    return {
        "mapping_cells": len(
            config["splits"]["mapping_pilot"]["cutoffs"]
        ),
        "shared_state_action_rounds": shared,
        "representative_probe_rounds": probes,
        "logical_state_action_rounds": logical,
        "fault_trajectories": trajectories,
        "fault_trajectory_rounds": trajectory_rounds,
        "total_formal_backend_rounds": (
            shared + probes + logical + trajectory_rounds
        ),
    }


def _base_report(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_config(config)
    intervals = {
        key: _interval(config["splits"][key], key)
        for key in (
            "formal_backend_a_seeds",
            "formal_backend_b_seeds",
            "trajectory_backend_a_seeds",
            "trajectory_backend_b_seeds",
        )
    }
    ranges = list(intervals.values())
    seed_disjoint = all(
        left[1] <= right[0] or right[1] <= left[0]
        for index, left in enumerate(ranges)
        for right in ranges[index + 1 :]
    )
    return {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "frozen_at": config["frozen_at"],
        "status": "PRE_FORMAL_IMMUTABLE_QUALIFICATION_SEAL",
        "formal_result_accessed": False,
        "config": {
            **_binding(
                root,
                config_path.relative_to(root).as_posix(),
            ),
            "semantic_sha256": _sha(
                _canonical(config).encode("utf-8")
            ),
        },
        "parents": {
            "backend_a": _released_parent(
                root,
                config["parent_a"],
                "T9.2.2",
            ),
            "backend_b": _released_parent(
                root,
                config["parent_b"],
                "T9.2.3",
            ),
            "action_contract": _released_parent(
                root,
                config["action_contract"],
                "T9.2.1",
            ),
        },
        "implementation": {
            "generator": _binding(
                root,
                "cnn_fpga/benchmark/phase9_twin_qualification_contract.py",
            ),
            "tests": _binding(
                root,
                "tests/test_phase9_twin_qualification_contract.py",
            ),
        },
        "seed_intervals": {
            key: {"start": value[0], "exclusive_stop": value[1]}
            for key, value in intervals.items()
        },
        "seed_intervals_pairwise_disjoint": seed_disjoint,
        "cell_accounting": _cell_accounting(config),
        "comparison_layers": {
            "shared_physical_state": (
                "identical Fock/qutrit density; compare transition, IQ, "
                "reset, leakage and action causality"
            ),
            "native_logical_channel": (
                "six native Pauli eigenstates per backend; compare logical "
                "PTM only after principal-angle mapping gate"
            ),
            "fault_trajectory": (
                "independent backend seeds; fixed step/telegraph/burst/"
                "compound interventions and complete denominator"
            ),
        },
        "prefrozen_tolerances": copy.deepcopy(
            config["prefrozen_tolerances"]
        ),
        "failure_policy": copy.deepcopy(config["failure_policy"]),
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
        "current_null_state": {
            key: None for key in config["claim_boundary"]["typed_null"]
        },
        "artifact_paths": copy.deepcopy(config["artifact_paths"]),
    }


def _evaluate(report: Mapping[str, Any], root: Path) -> dict[str, bool]:
    parents = report.get("parents", {})
    accounting = report.get("cell_accounting", {})
    policy = report.get("failure_policy", {})
    tolerances = report.get("prefrozen_tolerances", {})
    claims = report.get("current_null_state", {})
    return {
        "G01_preformal_identity": (
            report.get("task_id") == TASK_ID
            and report.get("status")
            == "PRE_FORMAL_IMMUTABLE_QUALIFICATION_SEAL"
            and report.get("formal_result_accessed") is False
        ),
        "G02_config_live_bound": (
            report.get("config")
            == {
                **_binding(
                    root,
                    "configs/phase9/t9_2_4_twin_qualification.json",
                ),
                "semantic_sha256": report.get("config", {}).get(
                    "semantic_sha256"
                ),
            }
            and len(str(report.get("config", {}).get("semantic_sha256", "")))
            == 64
        ),
        "G03_backend_a_bound": (
            parents.get("backend_a", {}).get("task_id") == "T9.2.2"
            and parents.get("backend_a", {}).get(
                "all_outcome_claims_null"
            )
            is True
        ),
        "G04_backend_b_bound": (
            parents.get("backend_b", {}).get("task_id") == "T9.2.3"
            and parents.get("backend_b", {}).get(
                "all_outcome_claims_null"
            )
            is True
        ),
        "G05_action_contract_bound": (
            parents.get("action_contract", {}).get("task_id") == "T9.2.1"
            and parents.get("action_contract", {}).get(
                "all_outcome_claims_null"
            )
            is True
        ),
        "G06_seed_intervals_disjoint": (
            report.get("seed_intervals_pairwise_disjoint") is True
            and len(report.get("seed_intervals", {})) == 4
        ),
        "G07_matrix_not_demo_sized": (
            accounting.get("shared_state_action_rounds") == 4480
            and accounting.get("representative_probe_rounds") == 512
            and accounting.get("logical_state_action_rounds") == 5376
            and accounting.get("fault_trajectories") == 384
            and accounting.get("fault_trajectory_rounds") == 4608
            and accounting.get("total_formal_backend_rounds") == 14976
        ),
        "G08_mapping_gate_frozen": (
            tolerances.get("minimum_code_principal_singular_value") == 0.95
            and tolerances.get("maximum_code_projector_frobenius") == 0.30
        ),
        "G09_physical_channel_tolerances_frozen": all(
            key in tolerances
            for key in (
                "maximum_ensemble_density_trace_distance",
                "maximum_mean_photon_difference",
                "maximum_level_probability_l1",
            )
        ),
        "G10_iq_tolerances_frozen": all(
            key in tolerances
            for key in (
                "maximum_integrated_iq_mean_difference",
                "maximum_integrated_iq_covariance_frobenius",
                "maximum_iq_two_sample_ks",
                "maximum_log_evidence_mean_difference",
                "maximum_posterior_mean_l1",
            )
        ),
        "G11_logical_tolerances_frozen": all(
            key in tolerances
            for key in (
                "maximum_logical_ptm_entry_difference",
                "maximum_logical_survival_difference",
            )
        ),
        "G12_tail_tolerances_frozen": all(
            key in tolerances
            for key in (
                "maximum_reset_success_rate_difference",
                "maximum_leakage_residence_rate_difference",
                "maximum_short_trajectory_terminal_trace_distance",
                "maximum_short_trajectory_observable_mean_difference",
            )
        ),
        "G13_simultaneous_error_control": (
            tolerances.get("familywise_alpha") == 0.05
            and tolerances.get("minimum_conservation_pass_fraction") == 1.0
        ),
        "G14_no_retuning": policy.get(
            "no_retuning_after_formal_access"
        )
        is True,
        "G15_no_mean_only_rescue": policy.get("no_mean_only_rescue") is True,
        "G16_no_cell_deletion": policy.get("no_cell_deletion") is True,
        "G17_no_codebook_claim": policy.get("no_codebook_claim") is True,
        "G18_fail_verdict_is_no_go": (
            policy.get("fail_verdict") == "NO_GO_TWIN_QUALIFICATION"
            and "prohibit T9.2.5"
            in policy.get("main_metric_failure", "")
        ),
        "G19_claims_typed_null": (
            len(claims) == 8
            and all(value is None for value in claims.values())
        ),
        "G20_implementation_live_bound": all(
            binding == _binding(root, str(binding["path"]))
            for binding in report.get("implementation", {}).values()
        ),
    }


Mutation = tuple[str, str, Callable[[dict[str, Any]], None]]


def _set(report: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _mutations() -> tuple[Mutation, ...]:
    return (
        ("M01", "G01_preformal_identity", lambda r: _set(r, ("formal_result_accessed",), True)),
        ("M02", "G02_config_live_bound", lambda r: _set(r, ("config", "bytes"), 0)),
        ("M03", "G03_backend_a_bound", lambda r: _set(r, ("parents", "backend_a", "task_id"), "T0")),
        ("M04", "G04_backend_b_bound", lambda r: _set(r, ("parents", "backend_b", "all_outcome_claims_null"), False)),
        ("M05", "G05_action_contract_bound", lambda r: _set(r, ("parents", "action_contract", "task_id"), "T0")),
        ("M06", "G06_seed_intervals_disjoint", lambda r: _set(r, ("seed_intervals_pairwise_disjoint",), False)),
        ("M07", "G07_matrix_not_demo_sized", lambda r: _set(r, ("cell_accounting", "total_formal_backend_rounds"), 10)),
        ("M08", "G08_mapping_gate_frozen", lambda r: _set(r, ("prefrozen_tolerances", "minimum_code_principal_singular_value"), 0.0)),
        ("M09", "G09_physical_channel_tolerances_frozen", lambda r: r["prefrozen_tolerances"].pop("maximum_ensemble_density_trace_distance")),
        ("M10", "G10_iq_tolerances_frozen", lambda r: r["prefrozen_tolerances"].pop("maximum_iq_two_sample_ks")),
        ("M11", "G11_logical_tolerances_frozen", lambda r: r["prefrozen_tolerances"].pop("maximum_logical_ptm_entry_difference")),
        ("M12", "G12_tail_tolerances_frozen", lambda r: r["prefrozen_tolerances"].pop("maximum_leakage_residence_rate_difference")),
        ("M13", "G13_simultaneous_error_control", lambda r: _set(r, ("prefrozen_tolerances", "familywise_alpha"), 0.5)),
        ("M14", "G14_no_retuning", lambda r: _set(r, ("failure_policy", "no_retuning_after_formal_access"), False)),
        ("M15", "G15_no_mean_only_rescue", lambda r: _set(r, ("failure_policy", "no_mean_only_rescue"), False)),
        ("M16", "G16_no_cell_deletion", lambda r: _set(r, ("failure_policy", "no_cell_deletion"), False)),
        ("M17", "G17_no_codebook_claim", lambda r: _set(r, ("failure_policy", "no_codebook_claim"), False)),
        ("M18", "G18_fail_verdict_is_no_go", lambda r: _set(r, ("failure_policy", "fail_verdict"), "PASS")),
        ("M19", "G19_claims_typed_null", lambda r: _set(r, ("current_null_state", "round_ler"), 0.0)),
        ("M20", "G20_implementation_live_bound", lambda r: _set(r, ("implementation", "generator", "sha256"), "0" * 64)),
    )


def build_report(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    active_root = _root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_4_twin_qualification.json"
        if config_path is None
        else config_path.resolve()
    )
    config = _load(active_config)
    report = _base_report(active_root, active_config, config)
    gates = _evaluate(report, active_root)
    report["gates"] = gates
    report["gate_summary"] = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "all_passed": all(gates.values()),
    }
    mutations: list[dict[str, Any]] = []
    for mutation_id, target, mutate in _mutations():
        changed = copy.deepcopy(report)
        mutate(changed)
        changed_gates = _evaluate(changed, active_root)
        mutations.append(
            {
                "mutation_id": mutation_id,
                "target_gate": target,
                "detected": changed_gates[target] is False,
            }
        )
    report["semantic_mutation_audit"] = mutations
    report["mutation_summary"] = {
        "detected": sum(row["detected"] for row in mutations),
        "total": len(mutations),
        "all_detected": all(row["detected"] for row in mutations),
    }
    report["verdict"] = (
        PASS_VERDICT
        if all(gates.values())
        and all(row["detected"] for row in mutations)
        else "NO_GO_T9_2_4_PREREGISTRATION"
    )
    analysis_payload = {
        key: value
        for key, value in report.items()
        if key not in {
            "gates",
            "gate_summary",
            "semantic_mutation_audit",
            "mutation_summary",
            "verdict",
            "analysis_sha256",
        }
    }
    report["analysis_sha256"] = _sha(
        _canonical(analysis_payload).encode("utf-8")
    )
    return report


def write_report(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    active_root = _root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_4_twin_qualification.json"
        if config_path is None
        else config_path.resolve()
    )
    config = _load(active_config)
    report = build_report(root=active_root, config_path=active_config)
    if report["verdict"] != PASS_VERDICT:
        raise RuntimeError("T9.2.4 preregistration failed closed")
    destination = active_root / config["artifact_paths"][
        "preregistration_seal"
    ]
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def verify_report(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, bool]:
    active_root = _root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_4_twin_qualification.json"
        if config_path is None
        else config_path.resolve()
    )
    config = _load(active_config)
    stored = _load(
        active_root / config["artifact_paths"]["preregistration_seal"]
    )
    rebuilt = build_report(root=active_root, config_path=active_config)
    return {
        "exact_rebuild": stored == rebuilt,
        "verdict": rebuilt["verdict"] == PASS_VERDICT,
        "gates": rebuilt["gate_summary"]["all_passed"] is True,
        "mutations": rebuilt["mutation_summary"]["all_detected"] is True,
        "formal_unaccessed": rebuilt["formal_result_accessed"] is False,
        "claims_null": all(
            value is None
            for value in rebuilt["current_null_state"].values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        print(write_report()["analysis_sha256"])
    if args.verify:
        checks = verify_report()
        print(json.dumps(checks, indent=2))
        return 0 if all(checks.values()) else 1
    if not args.write and not args.verify:
        print(json.dumps(build_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
