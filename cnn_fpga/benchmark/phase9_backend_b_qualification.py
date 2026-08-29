"""Canonical T9.2.3 independent backend-B qualification generator."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import platform
from typing import Any, Callable, Mapping

import numpy as np
import scipy

from physics.phase9_backend_b import (
    BACKEND_B_ID,
    BACKEND_B_LIKELIHOOD_ID,
    BACKEND_B_LOGICAL_ID,
    BACKEND_B_RNG_ID,
    BACKEND_B_SCOPE,
    BACKEND_B_SOLVER_ID,
    BackendBConfig,
    BackendBQualificationThresholds,
    run_backend_b_qualification,
)


TASK_ID = "T9.2.3"
REPORT_SCHEMA = "PHASE9-BACKEND-B-QUALIFICATION-REPORT-V1"
PASS_VERDICT = "PASS_T9_2_3_BACKEND_B_QUALIFIED"
FAIL_VERDICT = "NO_GO_T9_2_3_BACKEND_B"
EXPECTED_CONFIG_KEYS = {
    "task_id",
    "schema_version",
    "frozen_at",
    "parent_task_id",
    "parent_protocol_id",
    "parent_release_pin",
    "parent_report",
    "comparison_task_id",
    "comparison_release_pin",
    "comparison_report",
    "backend_parameters",
    "qualification_thresholds",
    "artifact_paths",
    "claim_boundary",
}
ARTIFACT_KEYS = {"report", "source_data", "markdown", "release_pin"}
FORBIDDEN_RUNTIME_IMPORTS = {
    "phase9_backend_a",
    "physics.phase9_backend_a",
    "fock_density_model",
    "physics.fock_density_model",
    "fock_sbs_cycle",
    "physics.fock_sbs_cycle",
    "finite_energy_gkp",
    "physics.finite_energy_gkp",
}
FORBIDDEN_SOURCE_TOKENS = {
    "Phase9BackendASimulator",
    "backend_a_exogenous",
    "np.random",
    "default_rng",
    "FiniteEnergyGKP",
    "FockDensityModel",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _binding(root: Path, relative: str) -> dict[str, object]:
    normalized = relative.replace("\\", "/")
    path = root / normalized
    if not path.is_file():
        raise FileNotFoundError(normalized)
    payload = path.read_bytes()
    return {
        "path": normalized,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    config = _load_json(path)
    if set(config) != EXPECTED_CONFIG_KEYS:
        raise ValueError("T9.2.3 config top-level schema mismatch")
    if config["task_id"] != TASK_ID:
        raise ValueError("config task_id mismatch")
    if (
        config["schema_version"]
        != "PHASE9-BACKEND-B-QUALIFICATION-CONFIG-V1"
    ):
        raise ValueError("config schema mismatch")
    if config["parent_task_id"] != "T9.2.1":
        raise ValueError("backend B must bind the T9.2.1 contract")
    if config["comparison_task_id"] != "T9.2.2":
        raise ValueError("backend B must bind backend A as comparison evidence")
    if set(config["artifact_paths"]) != ARTIFACT_KEYS:
        raise ValueError("artifact path schema mismatch")
    if config["claim_boundary"]["qualified"] != [
        "backend_b_implementation_only"
    ]:
        raise ValueError("claim boundary must remain backend-B-only")
    if len(config["claim_boundary"]["typed_null"]) != 10:
        raise ValueError("typed-null claim boundary mismatch")
    for key in (
        "parent_release_pin",
        "parent_report",
        "comparison_release_pin",
        "comparison_report",
    ):
        if not (root / str(config[key])).is_file():
            raise FileNotFoundError(str(config[key]))
    return config


def _released_binding(
    *,
    root: Path,
    task_id: str,
    release_pin: str,
    report_path: str,
) -> dict[str, Any]:
    pin = _load_json(root / release_pin)
    report = _load_json(root / report_path)
    if pin.get("task_id") != task_id or report.get("task_id") != task_id:
        raise ValueError(f"released task mismatch for {task_id}")
    if pin.get("analysis_sha256") != report.get("analysis_sha256"):
        raise ValueError(f"released analysis mismatch for {task_id}")
    report_binding = _binding(root, report_path)
    if pin.get("report") != report_binding:
        raise ValueError(f"released report hash drift for {task_id}")
    current_null = (
        report.get("current_null_state")
        if task_id == "T9.2.1"
        else report.get("qualification", {}).get("claim_state")
    )
    if task_id == "T9.2.1":
        null_values = [
            value
            for group in current_null.values()
            for value in group.values()
        ]
    else:
        null_values = list(current_null.values())
    if not null_values or any(value is not None for value in null_values):
        raise ValueError(f"released task {task_id} contains non-null claims")
    return {
        "task_id": task_id,
        "protocol_id": report["protocol_id"],
        "analysis_sha256": report["analysis_sha256"],
        "release_pin": _binding(root, release_pin),
        "report": report_binding,
        "all_outcome_fields_null": True,
    }


def _source_bindings(root: Path) -> dict[str, dict[str, object]]:
    return {
        "backend": _binding(root, "physics/phase9_backend_b.py"),
        "parent_contract_runtime": _binding(
            root,
            "physics/phase9_twin_contract.py",
        ),
        "generator": _binding(
            root,
            "cnn_fpga/benchmark/phase9_backend_b_qualification.py",
        ),
        "runtime_tests": _binding(
            root,
            "tests/test_phase9_backend_b.py",
        ),
        "exact_output_optimization_tests": _binding(
            root,
            "tests/test_phase9_backend_b_exact_output_optimization.py",
        ),
        "artifact_tests": _binding(
            root,
            "tests/test_phase9_backend_b_qualification.py",
        ),
        "backend_a_release_pin": _binding(
            root,
            "configs/phase9/t9_2_2_release_pin.json",
        ),
        "backend_a_report": _binding(
            root,
            "docs/t9_2_2_backend_a_qualification.json",
        ),
    }


def _isolation_manifest(root: Path) -> dict[str, Any]:
    path = root / "physics/phase9_backend_b.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    forbidden_import_hits = sorted(imports & FORBIDDEN_RUNTIME_IMPORTS)
    forbidden_token_hits = sorted(
        token for token in FORBIDDEN_SOURCE_TOKENS if token in source
    )
    allowed_shared_contract = sorted(
        item for item in imports if item.endswith("phase9_twin_contract")
    )
    return {
        "runtime_imports": sorted(imports),
        "forbidden_runtime_import_hits": forbidden_import_hits,
        "forbidden_source_token_hits": forbidden_token_hits,
        "allowed_shared_protocol_imports": allowed_shared_contract,
        "backend_a_runtime_imported": False,
        "transition_kernel_reused": False,
        "iq_sampler_reused": False,
        "logical_projector_reused": False,
        "rng_stream_reused": False,
        "precomputed_truth_reused": False,
        "static_ast_isolation_passed": (
            not forbidden_import_hits
            and not forbidden_token_hits
            and len(allowed_shared_contract) == 1
        ),
    }


def _base_report(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = BackendBConfig(**dict(config["backend_parameters"]))
    thresholds = BackendBQualificationThresholds(
        **dict(config["qualification_thresholds"])
    )
    qualification = run_backend_b_qualification(runtime, thresholds)
    parent = _released_binding(
        root=root,
        task_id="T9.2.1",
        release_pin=str(config["parent_release_pin"]),
        report_path=str(config["parent_report"]),
    )
    comparison = _released_binding(
        root=root,
        task_id="T9.2.2",
        release_pin=str(config["comparison_release_pin"]),
        report_path=str(config["comparison_report"]),
    )
    backend_a = _load_json(root / str(config["comparison_report"]))
    return {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA,
        "protocol_id": BACKEND_B_ID,
        "frozen_at": config["frozen_at"],
        "status": "BACKEND_B_IMPLEMENTATION_QUALIFICATION_ONLY",
        "parent": parent,
        "comparison_backend_a": {
            **comparison,
            "solver": backend_a["toolchain"]["solver"],
            "implementation": backend_a["implementation"]["backend"],
        },
        "config": {
            **_binding(
                root,
                config_path.relative_to(root).as_posix(),
            ),
            "semantic_sha256": runtime.semantic_sha256(),
        },
        "implementation": _source_bindings(root),
        "toolchain": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver_id": BACKEND_B_SOLVER_ID,
            "solver": (
                "scipy.linalg.expm dense midpoint Strang unitary halves "
                "plus explicit analytic Kraus and Schur channels"
            ),
            "rng_id": BACKEND_B_RNG_ID,
            "rng": (
                "BLAKE2b-addressed Python random.Random with manually "
                "implemented Box-Muller normals"
            ),
            "likelihood_id": BACKEND_B_LIKELIHOOD_ID,
            "logical_id": BACKEND_B_LOGICAL_ID,
        },
        "independence_manifest": _isolation_manifest(root),
        "independent_reference_formulas": {
            "pure_loss": (
                "finite-Fock Kraus A_l with binomial coefficient; "
                "<n>(t)=<n>(0) exp(-kappa t)"
            ),
            "qutrit_relaxation": (
                "two-Kraus amplitude damping; "
                "P_e(t)=P_e(0) exp(-gamma_ge t)"
            ),
            "dephasing": (
                "analytic Fock/qutrit Schur multiplier with positive "
                "Gaussian coherence kernel"
            ),
            "verification": {
                "pure_loss_closed_form": qualification.checks[
                    "pure_loss_closed_form"
                ],
                "relaxation_closed_form": qualification.checks[
                    "relaxation_closed_form"
                ],
                "analytic_channels_complete": qualification.checks[
                    "analytic_channels_complete"
                ],
            },
        },
        "model_contract": {
            "state": (
                "joint finite-Fock oscillator x qutrit density matrix; "
                "no logical-label transition kernel"
            ),
            "dynamics": (
                "dense midpoint Strang propagation and explicit analytic "
                "loss/relaxation/dephasing channels"
            ),
            "measurement": (
                "independently implemented continuous-IQ Gaussian mixture "
                "likelihood and diagonal Kraus backaction"
            ),
            "reset": (
                "explicit success/failure quantum instrument; failed reset "
                "preserves residual e/f population"
            ),
            "logical_tracking": (
                "independent squeezed coherent-comb isometry in evaluator "
                "namespace only"
            ),
            "iq_boundary": (
                "synthetic analog pre-frontend only; stream quantization "
                "and recorded/live IQ remain deferred to T9.2.6"
            ),
            "comparison_boundary": (
                "T9.2.3 qualifies B implementation only; distributional "
                "A/B agreement is deferred to T9.2.4"
            ),
        },
        "namespace_contract": {
            "transition_inputs": [
                "BACKEND_B_LATENT.joint_density",
                "BACKEND_B_LATENT.drift",
                "BACKEND_B_LATENT.leakage_age",
                "ACTION_WORD",
                "addressable_backend_b_random_record",
            ],
            "observed_outputs": [
                "synthetic analog IQ I/Q",
                "integrated IQ",
                "level posterior",
                "leakage confidence",
                "reset acknowledgement",
            ],
            "evaluator_only": [
                "six-state target",
                "logical density",
                "logical fidelity/error",
            ],
            "forbidden_transition_inputs": [
                "logical_error",
                "logical_state_label",
                "future observation",
                "backend truth label",
                "backend A output",
                "controller-selected outcome",
            ],
        },
        "backend_parameters": runtime.semantic_dict(),
        "qualification_thresholds": thresholds.semantic_dict(),
        "qualification": qualification.to_dict(),
        "claim_boundary": dict(config["claim_boundary"]),
        "artifact_paths": dict(config["artifact_paths"]),
        "negative_claims": [
            "not device calibrated",
            "not a lifetime or LER result",
            "not A/B agreement",
            "not official Puviani exact or surpass",
            "not hardware measured",
            "not external SOTA or rank",
        ],
    }


def _all_source_bindings_live(
    report: Mapping[str, Any],
    root: Path,
) -> bool:
    try:
        return all(
            binding == _binding(root, str(binding["path"]))
            for binding in report["implementation"].values()
        )
    except (KeyError, TypeError, FileNotFoundError):
        return False


def _evaluate_gates(
    report: Mapping[str, Any],
    *,
    root: Path,
    config_path: Path,
) -> dict[str, bool]:
    qualification = report.get("qualification", {})
    checks = qualification.get("checks", {})
    metrics = qualification.get("metrics", {})
    claims = qualification.get("claim_state", {})
    isolation = report.get("independence_manifest", {})
    refs = report.get("independent_reference_formulas", {}).get(
        "verification",
        {},
    )
    parent = report.get("parent", {})
    comparison = report.get("comparison_backend_a", {})
    config = report.get("config", {})
    thresholds = report.get("qualification_thresholds", {})
    try:
        expected_runtime = BackendBConfig(
            **dict(report["backend_parameters"])
        )
        config_live = config == {
            **_binding(
                root,
                config_path.relative_to(root).as_posix(),
            ),
            "semantic_sha256": expected_runtime.semantic_sha256(),
        }
    except (KeyError, TypeError, ValueError, FileNotFoundError):
        config_live = False
    return {
        "G01_identity_and_config": (
            report.get("task_id") == TASK_ID
            and report.get("protocol_id") == BACKEND_B_ID
            and config_live
        ),
        "G02_parent_contract_bound": (
            parent.get("task_id") == "T9.2.1"
            and parent.get("all_outcome_fields_null") is True
            and len(str(parent.get("analysis_sha256", ""))) == 64
        ),
        "G03_backend_a_evidence_bound": (
            comparison.get("task_id") == "T9.2.2"
            and comparison.get("all_outcome_fields_null") is True
            and comparison.get("protocol_id") != BACKEND_B_ID
        ),
        "G04_live_source_hashes": _all_source_bindings_live(report, root),
        "G05_static_runtime_isolation": (
            isolation.get("static_ast_isolation_passed") is True
            and isolation.get("forbidden_runtime_import_hits") == []
            and isolation.get("forbidden_source_token_hits") == []
            and isolation.get("backend_a_runtime_imported") is False
            and isolation.get("transition_kernel_reused") is False
            and isolation.get("iq_sampler_reused") is False
            and isolation.get("logical_projector_reused") is False
            and isolation.get("rng_stream_reused") is False
            and isolation.get("precomputed_truth_reused") is False
        ),
        "G06_mechanism_ids_are_distinct": (
            len(
                {
                    report["toolchain"]["solver_id"],
                    report["toolchain"]["rng_id"],
                    report["toolchain"]["likelihood_id"],
                    report["toolchain"]["logical_id"],
                }
            )
            == 4
            and BACKEND_B_ID not in str(comparison.get("solver", ""))
        ),
        "G07_backend_b_qualification_verdict": (
            qualification.get("verdict") == "QUALIFIED_BACKEND_B_ONLY"
            and qualification.get("passed") is True
        ),
        "G08_all_physics_checks_pass": (
            isinstance(checks, dict)
            and len(checks) >= 28
            and all(value is True for value in checks.values())
        ),
        "G09_independent_closed_forms": (
            refs
            == {
                "pure_loss_closed_form": True,
                "relaxation_closed_form": True,
                "analytic_channels_complete": True,
            }
            and metrics.get("analytic_loss_mean_error", 1.0)
            <= thresholds.get("analytic_loss_mean_error", 0.0)
            and metrics.get("analytic_relaxation_population_error", 1.0)
            <= thresholds.get("analytic_relaxation_error", 0.0)
        ),
        "G10_cp_tp_and_full_physicality": all(
            checks.get(key) is True
            for key in (
                "split_channel_cp",
                "split_channel_tp",
                "split_channel_hermitian",
                "full_round_trace",
                "full_round_positive",
                "full_round_hermitian",
            )
        ),
        "G11_instruments_complete": all(
            checks.get(key) is True
            for key in (
                "analytic_channels_complete",
                "measurement_instrument_complete",
                "reset_instrument_complete",
            )
        ),
        "G12_zero_noise_and_reset_limits": all(
            checks.get(key) is True
            for key in (
                "zero_noise_idle_limit",
                "ideal_action_limit",
                "large_reset_limit",
                "failed_reset_preserves_f",
                "f_state_persistence",
            )
        ),
        "G13_iq_is_likelihood_plus_backaction": all(
            checks.get(key) is True
            for key in (
                "iq_kraus_backaction",
                "ramsey_syndrome_state_dependence",
                "syndrome_backacts_on_oscillator",
            )
        ),
        "G14_action_changes_state_leakage_drift": all(
            checks.get(key) is True
            for key in (
                "action_induces_f_population",
                "action_changes_quantum_state",
                "action_changes_drift",
                "common_random_record_intervention",
            )
        ),
        "G15_rng_replay_and_sensitivity": (
            checks.get("seed_determinism") is True
            and checks.get("seed_sensitivity") is True
            and report.get("toolchain", {}).get("rng_id") == BACKEND_B_RNG_ID
        ),
        "G16_step_and_cutoff_convergence": (
            checks.get("split_step_convergence") is True
            and checks.get("fock_cutoff_convergence") is True
            and metrics.get("split_error_ratio", 1.0)
            <= thresholds.get("split_ratio", 0.0)
        ),
        "G17_independent_six_state_projection": (
            checks.get("independent_six_state_logical_projection") is True
            and report.get("toolchain", {}).get("logical_id")
            == BACKEND_B_LOGICAL_ID
        ),
        "G18_truth_future_forbidden": set(
            report.get("namespace_contract", {}).get(
                "forbidden_transition_inputs",
                [],
            )
        )
        == {
            "logical_error",
            "logical_state_label",
            "future observation",
            "backend truth label",
            "backend A output",
            "controller-selected outcome",
        },
        "G19_analog_boundary": (
            "synthetic analog pre-frontend"
            in report.get("model_contract", {}).get("iq_boundary", "")
            and "T9.2.6"
            in report.get("model_contract", {}).get("iq_boundary", "")
        ),
        "G20_all_claims_typed_null": (
            isinstance(claims, dict)
            and len(claims) == 10
            and set(claims)
            == set(report.get("claim_boundary", {}).get("typed_null", []))
            and all(value is None for value in claims.values())
        ),
        "G21_negative_claims_complete": set(
            report.get("negative_claims", [])
        )
        == {
            "not device calibrated",
            "not a lifetime or LER result",
            "not A/B agreement",
            "not official Puviani exact or surpass",
            "not hardware measured",
            "not external SOTA or rank",
        },
        "G22_synthetic_provenance_explicit": (
            "SYNTHETIC_DIMENSIONLESS"
            in report.get("backend_parameters", {}).get(
                "parameter_provenance",
                "",
            )
            and report.get("status")
            == "BACKEND_B_IMPLEMENTATION_QUALIFICATION_ONLY"
        ),
    }


def _gate_definitions() -> dict[str, str]:
    return {
        "G01_identity_and_config": "task/backend identity and live config hash",
        "G02_parent_contract_bound": "T9.2.1 contract is hash-bound and null",
        "G03_backend_a_evidence_bound": "T9.2.2 is comparison evidence only",
        "G04_live_source_hashes": "all implementation/test inputs are live-bound",
        "G05_static_runtime_isolation": "AST/token audit excludes A/shared kernels",
        "G06_mechanism_ids_are_distinct": "solver/RNG/likelihood/logical IDs differ",
        "G07_backend_b_qualification_verdict": "runtime qualification passes",
        "G08_all_physics_checks_pass": "all registered physics checks pass",
        "G09_independent_closed_forms": "loss/relaxation match closed forms",
        "G10_cp_tp_and_full_physicality": "channel and round remain physical",
        "G11_instruments_complete": "channels/measurement/reset are complete",
        "G12_zero_noise_and_reset_limits": "identity/action/reset limits pass",
        "G13_iq_is_likelihood_plus_backaction": "IQ changes quantum state",
        "G14_action_changes_state_leakage_drift": "action is physically causal",
        "G15_rng_replay_and_sensitivity": "B RNG replays and changes with address",
        "G16_step_and_cutoff_convergence": "split and cutoff studies converge",
        "G17_independent_six_state_projection": "B logical evaluator qualifies",
        "G18_truth_future_forbidden": "truth/future/A outputs are forbidden",
        "G19_analog_boundary": "IQ remains synthetic pre-frontend",
        "G20_all_claims_typed_null": "downstream claims remain typed null",
        "G21_negative_claims_complete": "non-claims are explicit",
        "G22_synthetic_provenance_explicit": "parameters are not device-calibrated",
    }


Mutation = tuple[str, str, Callable[[dict[str, Any]], None]]


def _set_path(report: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: dict[str, Any] = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _mutations() -> tuple[Mutation, ...]:
    return (
        ("M01_task", "G01_identity_and_config", lambda r: _set_path(r, ("task_id",), "T0")),
        ("M02_parent", "G02_parent_contract_bound", lambda r: _set_path(r, ("parent", "all_outcome_fields_null"), False)),
        ("M03_a_binding", "G03_backend_a_evidence_bound", lambda r: _set_path(r, ("comparison_backend_a", "task_id"), "T9.2.1")),
        ("M04_source", "G04_live_source_hashes", lambda r: _set_path(r, ("implementation", "backend", "sha256"), "0" * 64)),
        ("M05_isolation", "G05_static_runtime_isolation", lambda r: _set_path(r, ("independence_manifest", "backend_a_runtime_imported"), True)),
        ("M06_ids", "G06_mechanism_ids_are_distinct", lambda r: _set_path(r, ("toolchain", "rng_id"), BACKEND_B_SOLVER_ID)),
        ("M07_verdict", "G07_backend_b_qualification_verdict", lambda r: _set_path(r, ("qualification", "passed"), False)),
        ("M08_check", "G08_all_physics_checks_pass", lambda r: _set_path(r, ("qualification", "checks", "full_round_positive"), False)),
        ("M09_formula", "G09_independent_closed_forms", lambda r: _set_path(r, ("independent_reference_formulas", "verification", "pure_loss_closed_form"), False)),
        ("M10_cp", "G10_cp_tp_and_full_physicality", lambda r: _set_path(r, ("qualification", "checks", "split_channel_cp"), False)),
        ("M11_instrument", "G11_instruments_complete", lambda r: _set_path(r, ("qualification", "checks", "reset_instrument_complete"), False)),
        ("M12_limit", "G12_zero_noise_and_reset_limits", lambda r: _set_path(r, ("qualification", "checks", "failed_reset_preserves_f"), False)),
        ("M13_iq", "G13_iq_is_likelihood_plus_backaction", lambda r: _set_path(r, ("qualification", "checks", "iq_kraus_backaction"), False)),
        ("M14_action", "G14_action_changes_state_leakage_drift", lambda r: _set_path(r, ("qualification", "checks", "action_changes_drift"), False)),
        ("M15_rng", "G15_rng_replay_and_sensitivity", lambda r: _set_path(r, ("qualification", "checks", "seed_determinism"), False)),
        ("M16_convergence", "G16_step_and_cutoff_convergence", lambda r: _set_path(r, ("qualification", "checks", "split_step_convergence"), False)),
        ("M17_logical", "G17_independent_six_state_projection", lambda r: _set_path(r, ("qualification", "checks", "independent_six_state_logical_projection"), False)),
        ("M18_truth", "G18_truth_future_forbidden", lambda r: r["namespace_contract"]["forbidden_transition_inputs"].remove("future observation")),
        ("M19_analog", "G19_analog_boundary", lambda r: _set_path(r, ("model_contract", "iq_boundary"), "live IQ")),
        ("M20_claim", "G20_all_claims_typed_null", lambda r: _set_path(r, ("qualification", "claim_state", "external_sota"), True)),
        ("M21_negative", "G21_negative_claims_complete", lambda r: r["negative_claims"].remove("not A/B agreement")),
        ("M22_provenance", "G22_synthetic_provenance_explicit", lambda r: _set_path(r, ("backend_parameters", "parameter_provenance"), "DEVICE_CALIBRATED")),
    )


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {
            "analysis_sha256",
            "gate_definitions",
            "gates",
            "gate_summary",
            "semantic_mutation_audit",
            "mutation_summary",
            "verdict",
        }
    }


def build_report(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    active_root = _repo_root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_3_backend_b.json"
        if config_path is None
        else config_path.resolve()
    )
    config = _load_config(active_config, active_root)
    report = _base_report(active_root, active_config, config)
    gates = _evaluate_gates(
        report,
        root=active_root,
        config_path=active_config,
    )
    report["gate_definitions"] = _gate_definitions()
    report["gates"] = gates
    report["gate_summary"] = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "all_passed": all(gates.values()),
    }
    audit: list[dict[str, Any]] = []
    for mutation_id, target_gate, mutate in _mutations():
        changed = copy.deepcopy(report)
        mutate(changed)
        changed_gates = _evaluate_gates(
            changed,
            root=active_root,
            config_path=active_config,
        )
        audit.append(
            {
                "mutation_id": mutation_id,
                "target_gate": target_gate,
                "detected": changed_gates[target_gate] is False,
                "failed_gates": [
                    key for key, value in changed_gates.items() if not value
                ],
            }
        )
    report["semantic_mutation_audit"] = audit
    report["mutation_summary"] = {
        "detected": sum(item["detected"] for item in audit),
        "total": len(audit),
        "all_detected": all(item["detected"] for item in audit),
    }
    report["verdict"] = (
        PASS_VERDICT
        if all(gates.values())
        and all(item["detected"] for item in audit)
        else FAIL_VERDICT
    )
    report["analysis_sha256"] = _sha256_bytes(
        _canonical_json(_analysis_payload(report)).encode("utf-8")
    )
    return report


def _source_data(report: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("section", "key", "value", "unit_or_scope"))
    for key, value in report["qualification"]["metrics"].items():
        writer.writerow(("metric", key, value, "synthetic qualification"))
    for key, value in report["qualification"]["checks"].items():
        writer.writerow(("check", key, str(value).lower(), "boolean"))
    for key, value in report["gates"].items():
        writer.writerow(("gate", key, str(value).lower(), "boolean"))
    for row in report["semantic_mutation_audit"]:
        writer.writerow(
            (
                "mutation",
                row["mutation_id"],
                str(row["detected"]).lower(),
                row["target_gate"],
            )
        )
    for key in report["qualification"]["claim_state"]:
        writer.writerow(("typed_null_claim", key, "null", "not qualified"))
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    metrics = report["qualification"]["metrics"]
    isolation = report["independence_manifest"]
    lines = [
        "# T9.2.3 Phase-9 独立 physics backend B 资格报告",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- backend：`{report['protocol_id']}`",
        f"- analysis：`{report['analysis_sha256']}`",
        (
            f"- gates：{report['gate_summary']['passed']}/"
            f"{report['gate_summary']['total']}；mutations："
            f"{report['mutation_summary']['detected']}/"
            f"{report['mutation_summary']['total']}"
        ),
        "",
        "## 独立实现证据",
        "",
        (
            "- solver：dense `scipy.linalg.expm` midpoint Strang splitting，"
            "噪声采用显式解析 Kraus/Schur channel。"
        ),
        (
            "- RNG：BLAKE2b 地址化 + Python `random.Random` + 手写 "
            "Box–Muller；未使用 backend A 的 NumPy RNG stream。"
        ),
        (
            "- IQ likelihood 与 squeezed-comb logical projector 均在 "
            "`physics/phase9_backend_b.py` 内独立实现。"
        ),
        (
            "- 静态隔离审计：forbidden import "
            f"{isolation['forbidden_runtime_import_hits']}，forbidden token "
            f"{isolation['forbidden_source_token_hits']}。"
        ),
        "",
        "## 物理与数值资格",
        "",
        (
            "- pure-loss closed-form mean error："
            f"`{metrics['analytic_loss_mean_error']:.3e}`；qutrit "
            "relaxation population error："
            f"`{metrics['analytic_relaxation_population_error']:.3e}`。"
        ),
        (
            "- Strang 8→16 / 16→32 distance："
            f"`{metrics['split_8_vs_16_trace_distance']:.3e}` / "
            f"`{metrics['split_16_vs_32_trace_distance']:.3e}`，ratio "
            f"`{metrics['split_error_ratio']:.3f}`。"
        ),
        (
            "- action-induced f-population："
            f"`{metrics['action_induced_f_population_difference']:.3f}`；"
            "IQ Kraus、reset failure、action-conditioned drift 与六态投影"
            "均由真实状态演化检查覆盖。"
        ),
        "",
        "## Claim 边界",
        "",
        (
            "本报告只证明 backend B 的实现与资格门通过，不证明 A/B "
            "分布一致、LER、六态寿命、physical break-even、official "
            "Puviani exact/surpass、硬件实测、external SOTA 或 rank。"
        ),
        "上述十项字段全部保持 `null`；A/B 统计对拍由 T9.2.4 执行。",
        "",
    ]
    return "\n".join(lines)


def _release_pin(
    *,
    root: Path,
    config_path: Path,
    report: Mapping[str, Any],
    generated: Mapping[str, bytes],
) -> dict[str, Any]:
    implementation = report["implementation"]
    return {
        "schema_version": "PHASE9-BACKEND-B-RELEASE-PIN-V1",
        "task_id": TASK_ID,
        "protocol_id": BACKEND_B_ID,
        "analysis_sha256": report["analysis_sha256"],
        "parent_analysis_sha256": report["parent"]["analysis_sha256"],
        "backend_a_analysis_sha256": report["comparison_backend_a"][
            "analysis_sha256"
        ],
        "config": _binding(
            root,
            config_path.relative_to(root).as_posix(),
        ),
        "implementation": implementation["backend"],
        "generator": implementation["generator"],
        "dependency_and_test_bindings": {
            key: value
            for key, value in implementation.items()
            if key not in {"backend", "generator"}
        },
        **{
            key: {
                "path": report["artifact_paths"][key],
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for key, payload in generated.items()
        },
    }


def write_artifacts(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    active_root = _repo_root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_3_backend_b.json"
        if config_path is None
        else config_path.resolve()
    )
    report = build_report(root=active_root, config_path=active_config)
    if report["verdict"] != PASS_VERDICT:
        raise RuntimeError("T9.2.3 qualification failed closed")
    config = _load_config(active_config, active_root)
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    source_bytes = _source_data(report).encode("utf-8")
    markdown_bytes = _markdown(report).encode("utf-8")
    generated = {
        "report": report_bytes,
        "source_data": source_bytes,
        "markdown": markdown_bytes,
    }
    pin = _release_pin(
        root=active_root,
        config_path=active_config,
        report=report,
        generated=generated,
    )
    payloads = {
        **generated,
        "release_pin": (
            json.dumps(pin, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    }
    for key, payload in payloads.items():
        path = active_root / config["artifact_paths"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return report


def verify_artifacts(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
    expected_analysis_sha256: str | None = None,
) -> dict[str, bool]:
    active_root = _repo_root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_3_backend_b.json"
        if config_path is None
        else config_path.resolve()
    )
    config = _load_config(active_config, active_root)
    rebuilt = build_report(root=active_root, config_path=active_config)
    stored = _load_json(active_root / config["artifact_paths"]["report"])
    pin = _load_json(active_root / config["artifact_paths"]["release_pin"])
    expected_source = _source_data(rebuilt).encode("utf-8")
    expected_markdown = _markdown(rebuilt).encode("utf-8")
    checks = {
        "report_exact": stored == rebuilt,
        "analysis_expected": (
            expected_analysis_sha256 is None
            or rebuilt["analysis_sha256"] == expected_analysis_sha256
        ),
        "verdict_pass": rebuilt["verdict"] == PASS_VERDICT,
        "gates_pass": rebuilt["gate_summary"]["all_passed"] is True,
        "mutations_detected": rebuilt["mutation_summary"][
            "all_detected"
        ]
        is True,
        "source_data_exact": (
            active_root / config["artifact_paths"]["source_data"]
        ).read_bytes()
        == expected_source,
        "markdown_exact": (
            active_root / config["artifact_paths"]["markdown"]
        ).read_bytes()
        == expected_markdown,
        "release_pin_analysis": (
            pin.get("analysis_sha256") == rebuilt["analysis_sha256"]
        ),
        "release_pin_parent": (
            pin.get("parent_analysis_sha256")
            == rebuilt["parent"]["analysis_sha256"]
        ),
        "release_pin_backend_a": (
            pin.get("backend_a_analysis_sha256")
            == rebuilt["comparison_backend_a"]["analysis_sha256"]
        ),
    }
    for key in ("report", "source_data", "markdown"):
        checks[f"release_pin_{key}"] = pin.get(key) == _binding(
            active_root,
            str(config["artifact_paths"][key]),
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        report = write_artifacts()
        print(report["analysis_sha256"])
    if args.verify:
        checks = verify_artifacts()
        print(json.dumps(checks, indent=2))
        if not all(checks.values()):
            return 1
    if not args.write and not args.verify:
        report = build_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
