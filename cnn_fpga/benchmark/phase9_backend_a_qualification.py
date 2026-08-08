"""Canonical T9.2.2 backend-A qualification artifact generator."""

from __future__ import annotations

import argparse
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

from physics.phase9_backend_a import (
    BACKEND_A_ID,
    BACKEND_A_SCOPE,
    BackendAConfig,
    BackendAQualificationThresholds,
    run_backend_a_qualification,
)


TASK_ID = "T9.2.2"
REPORT_SCHEMA = "PHASE9-BACKEND-A-QUALIFICATION-REPORT-V1"
PROTOCOL_ID = "PHASE9-BACKEND-A-JOINT-FOCK-QUTRIT-GKSL-V1"
PASS_VERDICT = "QUALIFIED_BACKEND_A_ONLY"
FAIL_VERDICT = "NO_GO_BACKEND_A_QUALIFICATION"

REQUIRED_CONFIG_KEYS = {
    "task_id",
    "schema_version",
    "frozen_at",
    "parent_task_id",
    "parent_protocol_id",
    "parent_release_pin",
    "parent_report",
    "backend_parameters",
    "qualification_thresholds",
    "artifact_paths",
    "claim_boundary",
}
REQUIRED_ARTIFACT_KEYS = {
    "report",
    "source_data",
    "markdown",
    "release_pin",
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


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _binding(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    payload = path.read_bytes()
    return {
        "path": relative.replace("\\", "/"),
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
    if set(config) != REQUIRED_CONFIG_KEYS:
        raise ValueError("T9.2.2 config top-level schema mismatch")
    if config["task_id"] != TASK_ID:
        raise ValueError("config task_id mismatch")
    if config["schema_version"] != "PHASE9-BACKEND-A-QUALIFICATION-CONFIG-V1":
        raise ValueError("config schema version mismatch")
    if config["parent_task_id"] != "T9.2.1":
        raise ValueError("T9.2.2 must bind T9.2.1")
    if config["parent_protocol_id"] != "PHASE9-CAUSAL-TWIN-CONTRACT-V1":
        raise ValueError("parent protocol mismatch")
    if set(config["artifact_paths"]) != REQUIRED_ARTIFACT_KEYS:
        raise ValueError("artifact path schema mismatch")
    for relative in (
        config["parent_release_pin"],
        config["parent_report"],
    ):
        if not (root / relative).is_file():
            raise FileNotFoundError(relative)
    qualified = config["claim_boundary"]["qualified"]
    typed_null = config["claim_boundary"]["typed_null"]
    if qualified != ["backend_a_implementation_only"]:
        raise ValueError("qualification boundary must remain backend-A-only")
    if not isinstance(typed_null, list) or len(typed_null) != 10:
        raise ValueError("typed-null claim boundary mismatch")
    return config


def _backend_config(config: Mapping[str, Any]) -> BackendAConfig:
    return BackendAConfig(**dict(config["backend_parameters"]))


def _thresholds(
    config: Mapping[str, Any],
) -> BackendAQualificationThresholds:
    return BackendAQualificationThresholds(
        **dict(config["qualification_thresholds"])
    )


def _parent_binding(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pin_relative = str(config["parent_release_pin"])
    report_relative = str(config["parent_report"])
    pin = _load_json(root / pin_relative)
    report = _load_json(root / report_relative)
    if pin.get("task_id") != "T9.2.1":
        raise ValueError("parent release pin task mismatch")
    if pin.get("protocol_id") != config["parent_protocol_id"]:
        raise ValueError("parent release pin protocol mismatch")
    if report.get("task_id") != "T9.2.1":
        raise ValueError("parent report task mismatch")
    if report.get("protocol_id") != config["parent_protocol_id"]:
        raise ValueError("parent report protocol mismatch")
    if report.get("analysis_sha256") != pin.get("analysis_sha256"):
        raise ValueError("parent analysis does not match release pin")
    for name in (
        "config",
        "implementation",
        "physics_contract",
        "report",
        "totality_manifest",
        "source_data",
        "markdown",
    ):
        expected = pin.get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"parent release pin lacks {name}")
        actual = _binding(root, str(expected["path"]))
        if actual != expected:
            raise ValueError(f"parent binding drift: {name}")
    current_null_state = report.get("current_null_state")
    if not isinstance(current_null_state, dict):
        raise ValueError("parent report lacks current_null_state")
    for group in current_null_state.values():
        if not isinstance(group, dict) or any(
            value is not None for value in group.values()
        ):
            raise ValueError("parent report contains a non-null outcome")
    return {
        "task_id": "T9.2.1",
        "protocol_id": config["parent_protocol_id"],
        "analysis_sha256": pin["analysis_sha256"],
        "release_pin": _binding(root, pin_relative),
        "report": _binding(root, report_relative),
        "all_parent_outcome_fields_null": True,
    }


def _source_binding(root: Path) -> dict[str, dict[str, object]]:
    return {
        "backend": _binding(root, "physics/phase9_backend_a.py"),
        "fock_density_dependency": _binding(
            root,
            "physics/fock_density_model.py",
        ),
        "fock_sbs_dependency": _binding(
            root,
            "physics/fock_sbs_cycle.py",
        ),
        "finite_gkp_dependency": _binding(
            root,
            "physics/finite_energy_gkp.py",
        ),
        "quadrature_dependency": _binding(
            root,
            "physics/quadrature_conventions.py",
        ),
        "sbs_error_space_dependency": _binding(
            root,
            "physics/sbs_error_space.py",
        ),
        "parent_contract_runtime": _binding(
            root,
            "physics/phase9_twin_contract.py",
        ),
        "generator": _binding(
            root,
            "cnn_fpga/benchmark/phase9_backend_a_qualification.py",
        ),
        "runtime_tests": _binding(
            root,
            "tests/test_phase9_backend_a.py",
        ),
        "artifact_tests": _binding(
            root,
            "tests/test_phase9_backend_a_qualification.py",
        ),
    }


def _base_report(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    backend_config = _backend_config(config)
    thresholds = _thresholds(config)
    qualification = run_backend_a_qualification(
        backend_config,
        thresholds,
    )
    return {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "frozen_at": config["frozen_at"],
        "status": "BACKEND_A_IMPLEMENTATION_QUALIFICATION_ONLY",
        "parent": _parent_binding(root, config),
        "config": {
            **_binding(
                root,
                config_path.relative_to(root).as_posix(),
            ),
            "semantic_sha256": backend_config.semantic_sha256(),
        },
        "implementation": _source_binding(root),
        "toolchain": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": (
                "scipy.sparse.linalg.expm_multiply for piecewise GKSL; "
                "scipy.linalg.expm for small-system Choi audit"
            ),
            "vectorization": "column-major vec(A rho B)=(B^T kron A)vec(rho)",
        },
        "model_contract": {
            "state": "joint density matrix on finite Fock oscillator x qutrit {g,e,f}",
            "dynamics": (
                "piecewise time-dependent Hamiltonian plus GKSL collapse "
                "operators; no logical-label transition kernel"
            ),
            "measurement": (
                "Ramsey-like qutrit interaction followed by continuous-IQ "
                "diagonal Kraus likelihood and conditional backaction"
            ),
            "reset": (
                "success/failure quantum instrument; failed branch preserves "
                "e/f population and success branch maps g/e/f to g"
            ),
            "drift": (
                "five-dimensional latent AR recurrence with action energy and "
                "Pauli primitive kicks plus addressable exogenous noise"
            ),
            "logical_tracking": (
                "partial trace and finite-GKP code projection in evaluator "
                "namespace only; evaluator cannot affect transition or IQ"
            ),
            "action_source": (
                "semantic+CRC-valid T9.2.1 ActionWord representative probes; "
                "pre-codebook/non-ranking"
            ),
            "iq_boundary": (
                "synthetic analog pre-frontend only; digitization/Q-format/"
                "stream semantics deferred to T9.2.6"
            ),
        },
        "namespace_contract": {
            "transition_inputs": [
                "BACKEND_LATENT.joint_density",
                "BACKEND_LATENT.drift",
                "BACKEND_LATENT.leakage_age",
                "ACTION_WORD",
                "addressable_exogenous_record",
            ],
            "observed_outputs": [
                "analog IQ I/Q",
                "integrated IQ",
                "level posterior",
                "leakage confidence",
                "reset acknowledgement",
            ],
            "truth_outputs": [
                "sampled emission component",
                "hidden reset outcome",
                "level populations",
                "drift before/after",
                "density diagnostics",
            ],
            "forbidden_transition_inputs": [
                "logical_error",
                "logical_state_label",
                "future observation",
                "backend truth label",
                "controller-selected outcome",
            ],
        },
        "backend_parameters": backend_config.semantic_dict(),
        "qualification_thresholds": thresholds.semantic_dict(),
        "qualification": qualification.to_dict(),
        "claim_boundary": {
            "qualified": ["backend_a_implementation_only"],
            "not_qualified": [
                "backend B",
                "dual-backend agreement",
                "codebook",
                "fixed-point frontend",
                "LER/lifetime",
                "physical break-even",
                "hardware/HIL",
                "official Puviani",
                "external SOTA/rank",
            ],
        },
        "artifact_paths": dict(config["artifact_paths"]),
    }


def _gate_definitions() -> list[dict[str, str]]:
    return [
        {"gate_id": "G01_identity", "description": "task/protocol identity exact"},
        {"gate_id": "G02_parent_live", "description": "T9.2.1 live release binding"},
        {"gate_id": "G03_config_live", "description": "config bytes and semantic hash live"},
        {"gate_id": "G04_source_live", "description": "backend/generator/parent runtime hashes live"},
        {"gate_id": "G05_backend_verdict", "description": "backend-A-only PASS"},
        {"gate_id": "G06_all_physics_checks", "description": "all registered checks true"},
        {"gate_id": "G07_cp_tp", "description": "actual GKSL Choi CP/TP/Hermitian"},
        {"gate_id": "G08_density_physical", "description": "full-round density physicality"},
        {"gate_id": "G09_instruments_complete", "description": "IQ/reset instruments complete"},
        {"gate_id": "G10_physical_limits", "description": "zero/ideal/reset limits"},
        {"gate_id": "G11_iq_backaction", "description": "IQ likelihood changes density"},
        {"gate_id": "G12_leakage_reset", "description": "reset failure and f persistence"},
        {"gate_id": "G13_action_quantum", "description": "action changes joint quantum transition"},
        {"gate_id": "G14_action_drift", "description": "action changes latent drift"},
        {"gate_id": "G15_rng_causal", "description": "CRN, determinism and stochastic sensitivity"},
        {"gate_id": "G16_convergence", "description": "step-size and Fock-cutoff convergence"},
        {"gate_id": "G17_logical_evaluator", "description": "six-state evaluator projection"},
        {"gate_id": "G18_analog_boundary", "description": "IQ remains synthetic analog pre-frontend"},
        {"gate_id": "G19_claims_null", "description": "all downstream claim fields typed null"},
        {"gate_id": "G20_backend_a_only", "description": "no backend-B/dual-backend promotion"},
    ]


def _evaluate_gates(
    report: Mapping[str, Any],
    *,
    root: Path,
    config_path: Path,
) -> dict[str, bool]:
    qualification = report["qualification"]
    checks = qualification["checks"]
    metrics = qualification["metrics"]
    parent = report["parent"]
    config = report["config"]
    implementation = report["implementation"]
    live_parent_pin = _binding(root, str(parent["release_pin"]["path"]))
    live_parent_report = _binding(root, str(parent["report"]["path"]))
    live_parent_pin_payload = _load_json(
        root / str(parent["release_pin"]["path"])
    )
    live_config_payload = _load_json(config_path)
    live_sources = _source_binding(root)
    live_config = _binding(root, config_path.relative_to(root).as_posix())
    thresholds = report["qualification_thresholds"]
    claim_state = qualification["claim_state"]
    return {
        "G01_identity": (
            report["task_id"] == TASK_ID
            and report["schema_version"] == REPORT_SCHEMA
            and report["protocol_id"] == PROTOCOL_ID
        ),
        "G02_parent_live": (
            parent["task_id"] == "T9.2.1"
            and parent["protocol_id"] == "PHASE9-CAUSAL-TWIN-CONTRACT-V1"
            and parent["release_pin"] == live_parent_pin
            and parent["report"] == live_parent_report
            and parent["analysis_sha256"]
            == live_parent_pin_payload["analysis_sha256"]
            and parent["all_parent_outcome_fields_null"] is True
        ),
        "G03_config_live": (
            {key: config[key] for key in ("path", "bytes", "sha256")}
            == live_config
            and config["semantic_sha256"]
            == BackendAConfig(**dict(report["backend_parameters"])).semantic_sha256()
            and report["backend_parameters"]
            == live_config_payload["backend_parameters"]
            and report["qualification_thresholds"]
            == live_config_payload["qualification_thresholds"]
            and report["artifact_paths"]
            == live_config_payload["artifact_paths"]
        ),
        "G04_source_live": implementation == live_sources,
        "G05_backend_verdict": (
            qualification["verdict"] == PASS_VERDICT
            and qualification["passed"] is True
        ),
        "G06_all_physics_checks": (
            len(checks) >= 27 and all(checks.values())
        ),
        "G07_cp_tp": all(
            checks[name]
            for name in (
                "gksl_channel_cp",
                "gksl_channel_tp",
                "gksl_choi_hermitian",
            )
        ),
        "G08_density_physical": all(
            checks[name]
            for name in (
                "full_round_trace",
                "full_round_hermiticity",
                "full_round_positive",
                "probabilities_normalized",
            )
        ),
        "G09_instruments_complete": (
            checks["measurement_instrument_complete"]
            and checks["reset_instrument_complete"]
        ),
        "G10_physical_limits": all(
            checks[name]
            for name in (
                "zero_noise_idle_limit",
                "ideal_action_limit",
                "large_reset_limit",
            )
        ),
        "G11_iq_backaction": (
            checks["iq_drives_measurement_backaction"]
            and checks["ramsey_syndrome_state_dependence"]
            and checks["syndrome_measurement_backacts_on_oscillator"]
            and metrics["measurement_posterior_peak"]
            > thresholds["measurement_posterior_peak"]
            and metrics["syndrome_fock0_vs_fock1_level_tv"]
            > thresholds["syndrome_state_dependence_minimum"]
            and metrics["syndrome_oscillator_backaction_trace_distance"]
            > thresholds["syndrome_backaction_trace_distance_minimum"]
        ),
        "G12_leakage_reset": all(
            checks[name]
            for name in (
                "reset_failure_preserves_f",
                "f_state_persistence",
                "action_induces_physical_f_population",
            )
        )
        and metrics["action_induced_f_population_difference"]
        > thresholds["action_induced_f_population_minimum"],
        "G13_action_quantum": (
            checks["action_changes_quantum_transition"]
            and metrics["action_intervention_state_trace_distance"]
            > thresholds["action_state_trace_distance_minimum"]
        ),
        "G14_action_drift": (
            checks["action_changes_latent_drift"]
            and metrics["action_intervention_drift_l2"]
            > thresholds["action_drift_l2_minimum"]
        ),
        "G15_rng_causal": all(
            checks[name]
            for name in (
                "intervention_uses_common_randomness",
                "seed_determinism",
                "different_seed_changes_observation",
            )
        ),
        "G16_convergence": (
            checks["step_size_convergence"]
            and checks["fock_cutoff_convergence"]
            and metrics["step_size_16_vs_32_trace_distance"]
            <= thresholds["step_size_trace_distance"]
            and metrics["step_size_error_ratio"]
            <= thresholds["step_size_error_ratio"]
            and metrics["fock_cutoff_8_vs_12_trace_distance"]
            <= thresholds["fock_cutoff_trace_distance"]
        ),
        "G17_logical_evaluator": (
            checks["logical_tracking_finite"]
            and checks["six_state_logical_projection"]
        ),
        "G18_analog_boundary": (
            report["model_contract"]["iq_boundary"]
            == (
                "synthetic analog pre-frontend only; digitization/Q-format/"
                "stream semantics deferred to T9.2.6"
            )
            and not any(
                token in str(item).lower()
                for item in report["namespace_contract"]["observed_outputs"]
                for token in ("live", "recorded", "fixed-point", "digitized")
            )
        ),
        "G19_claims_null": (
            len(claim_state) == 10
            and set(claim_state)
            == set(live_config_payload["claim_boundary"]["typed_null"])
            and all(value is None for value in claim_state.values())
        ),
        "G20_backend_a_only": (
            report["claim_boundary"]["qualified"]
            == ["backend_a_implementation_only"]
            and "backend B" in report["claim_boundary"]["not_qualified"]
            and claim_state["backend_b_qualified"] is None
            and claim_state["dual_backend_agreement"] is None
        ),
    }


def _mutations() -> list[
    tuple[str, str, Callable[[dict[str, Any]], None]]
]:
    def set_path(*path_and_value: object) -> Callable[[dict[str, Any]], None]:
        *path, value = path_and_value

        def mutate(report: dict[str, Any]) -> None:
            target: Any = report
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value

        return mutate

    return [
        ("M01_identity", "G01_identity", set_path("task_id", "T9.2.X")),
        (
            "M02_parent_hash",
            "G02_parent_live",
            set_path("parent", "release_pin", "sha256", "0" * 64),
        ),
        (
            "M03_config_hash",
            "G03_config_live",
            set_path("config", "semantic_sha256", "0" * 64),
        ),
        (
            "M04_source_hash",
            "G04_source_live",
            set_path(
                "implementation",
                "backend",
                "sha256",
                "0" * 64,
            ),
        ),
        (
            "M05_verdict",
            "G05_backend_verdict",
            set_path("qualification", "verdict", FAIL_VERDICT),
        ),
        (
            "M06_check",
            "G06_all_physics_checks",
            set_path(
                "qualification",
                "checks",
                "seed_determinism",
                False,
            ),
        ),
        (
            "M07_cp",
            "G07_cp_tp",
            set_path(
                "qualification",
                "checks",
                "gksl_channel_cp",
                False,
            ),
        ),
        (
            "M08_density",
            "G08_density_physical",
            set_path(
                "qualification",
                "checks",
                "full_round_positive",
                False,
            ),
        ),
        (
            "M09_reset_complete",
            "G09_instruments_complete",
            set_path(
                "qualification",
                "checks",
                "reset_instrument_complete",
                False,
            ),
        ),
        (
            "M10_limit",
            "G10_physical_limits",
            set_path(
                "qualification",
                "checks",
                "large_reset_limit",
                False,
            ),
        ),
        (
            "M11_iq_label_only",
            "G11_iq_backaction",
            set_path(
                "qualification",
                "checks",
                "iq_drives_measurement_backaction",
                False,
            ),
        ),
        (
            "M12_f_persistence",
            "G12_leakage_reset",
            set_path(
                "qualification",
                "checks",
                "f_state_persistence",
                False,
            ),
        ),
        (
            "M13_action_label_only",
            "G13_action_quantum",
            set_path(
                "qualification",
                "metrics",
                "action_intervention_state_trace_distance",
                0.0,
            ),
        ),
        (
            "M14_action_no_drift",
            "G14_action_drift",
            set_path(
                "qualification",
                "metrics",
                "action_intervention_drift_l2",
                0.0,
            ),
        ),
        (
            "M15_no_crn",
            "G15_rng_causal",
            set_path(
                "qualification",
                "checks",
                "intervention_uses_common_randomness",
                False,
            ),
        ),
        (
            "M16_no_convergence",
            "G16_convergence",
            set_path(
                "qualification",
                "metrics",
                "step_size_error_ratio",
                1.0,
            ),
        ),
        (
            "M17_no_six_state",
            "G17_logical_evaluator",
            set_path(
                "qualification",
                "checks",
                "six_state_logical_projection",
                False,
            ),
        ),
        (
            "M18_relabel_live",
            "G18_analog_boundary",
            set_path("model_contract", "iq_boundary", "live IQ qualified"),
        ),
        (
            "M19_fill_claim",
            "G19_claims_null",
            set_path(
                "qualification",
                "claim_state",
                "external_sota",
                True,
            ),
        ),
        (
            "M20_dual_promote",
            "G20_backend_a_only",
            set_path(
                "claim_boundary",
                "qualified",
                ["backend_a_implementation_only", "dual_backend_agreement"],
            ),
        ),
    ]


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {
            "analysis_sha256",
            "gates",
            "gate_summary",
            "semantic_mutation_audit",
        }
    }


def build_report(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    active_root = _repo_root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_2_backend_a.json"
        if config_path is None
        else config_path.resolve()
    )
    config = _load_config(active_config, active_root)
    report = _base_report(active_root, active_config, config)
    definitions = _gate_definitions()
    gates = _evaluate_gates(
        report,
        root=active_root,
        config_path=active_config,
    )
    report["gate_definitions"] = definitions
    report["gates"] = gates
    report["gate_summary"] = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "all_passed": all(gates.values()),
    }
    audit: list[dict[str, Any]] = []
    for mutation_id, target_gate, mutate in _mutations():
        mutated = copy.deepcopy(report)
        mutate(mutated)
        mutated_gates = _evaluate_gates(
            mutated,
            root=active_root,
            config_path=active_config,
        )
        detected = mutated_gates[target_gate] is False
        audit.append(
            {
                "mutation_id": mutation_id,
                "target_gate": target_gate,
                "detected": detected,
                "failed_gates": [
                    key for key, value in mutated_gates.items() if not value
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
        "PASS_T9_2_2_BACKEND_A_QUALIFIED"
        if all(gates.values())
        and all(item["detected"] for item in audit)
        else "NO_GO_T9_2_2_BACKEND_A"
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
    for item in report["semantic_mutation_audit"]:
        writer.writerow(
            (
                "mutation",
                item["mutation_id"],
                str(item["detected"]).lower(),
                item["target_gate"],
            )
        )
    for key, value in report["qualification"]["claim_state"].items():
        writer.writerow(("typed_null_claim", key, "null", "not qualified"))
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    metrics = report["qualification"]["metrics"]
    lines = [
        "# T9.2.2 Phase-9 高保真 physics backend A 资格报告",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- backend：`{report['protocol_id']}`",
        f"- analysis：`{report['analysis_sha256']}`",
        f"- parent T9.2.1：`{report['parent']['analysis_sha256']}`",
        (
            f"- gates：{report['gate_summary']['passed']}/"
            f"{report['gate_summary']['total']}；mutations："
            f"{report['mutation_summary']['detected']}/"
            f"{report['mutation_summary']['total']}"
        ),
        "",
        "## 实际模型",
        "",
        (
            "backend A 逐轮维护有限 Fock oscillator × `g/e/f` qutrit 的联合密度矩阵。"
            "恢复动作进入时间依赖 Hamiltonian；loss/dephasing/relaxation/excitation "
            "进入 GKSL generator；Ramsey-like interaction 后由连续 IQ likelihood "
            "构造对角 Kraus 并真实更新密度矩阵。reset 是 success/failure quantum "
            "instrument，失败支路保留 `e/f`，不是给标签加独立噪声。"
        ),
        "",
        (
            "动作还进入五维 latent drift 递推；共同 exogenous record 下，IDLE/X "
            f"的 joint-state trace distance=`{metrics['action_intervention_state_trace_distance']:.6g}`，"
            f"drift L2=`{metrics['action_intervention_drift_l2']:.6g}`。"
        ),
        "",
        "## 物理性、极限与收敛",
        "",
        (
            f"- Choi 最小特征值 `{metrics['choi_minimum_eigenvalue']:.6g}`，"
            f"TP Frobenius `{metrics['choi_tp_frobenius']:.6g}`；"
        ),
        (
            f"- full-round 最小特征值 `{metrics['full_round_minimum_eigenvalue']:.6g}`，"
            f"trace error `{metrics['full_round_trace_error']:.6g}`；"
        ),
        (
            f"- ideal action distance `{metrics['ideal_action_trace_distance']:.6g}`，"
            f"zero-noise idle distance `{metrics['zero_noise_idle_trace_distance']:.6g}`；"
        ),
        (
            f"- step 8→16 / 16→32 distance "
            f"`{metrics['step_size_8_vs_16_trace_distance']:.6g}` / "
            f"`{metrics['step_size_16_vs_32_trace_distance']:.6g}`，"
            f"ratio `{metrics['step_size_error_ratio']:.6g}`；"
        ),
        (
            f"- cutoff 8→12 distance "
            f"`{metrics['fock_cutoff_8_vs_12_trace_distance']:.6g}`；"
        ),
        (
            f"- large-reset `g` probability `{metrics['large_reset_g_probability']:.12g}`，"
            f"failed-reset `f` persistence `{metrics['failed_reset_f_probability']:.12g}`。"
        ),
        "",
        "## 证据边界",
        "",
        (
            "本 PASS 只资格化 backend A 的实现。IQ 是 synthetic analog pre-frontend；"
            "T9.2.6 前不具备 ADC/Q-format/stream 资格。backend B、双后端对拍、"
            "codebook、LER、六态 lifetime、physical break-even、硬件/HIL、"
            "official Puviani、external SOTA 与 rank 全部保持 `null`。"
        ),
        "",
    ]
    return "\n".join(lines)


def _release_pin(
    *,
    root: Path,
    config_path: Path,
    report: Mapping[str, Any],
    source_data: str,
    markdown: str,
) -> dict[str, Any]:
    paths = report["artifact_paths"]
    report_bytes = (
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    source_bytes = source_data.encode("utf-8")
    markdown_bytes = markdown.encode("utf-8")
    return {
        "schema_version": "PHASE9-BACKEND-A-RELEASE-PIN-V1",
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "analysis_sha256": report["analysis_sha256"],
        "parent_analysis_sha256": report["parent"]["analysis_sha256"],
        "config": _binding(
            root,
            config_path.relative_to(root).as_posix(),
        ),
        "implementation": report["implementation"]["backend"],
        "generator": report["implementation"]["generator"],
        "dependency_and_test_bindings": {
            key: value
            for key, value in report["implementation"].items()
            if key not in {"backend", "generator"}
        },
        "report": {
            "path": paths["report"],
            "bytes": len(report_bytes),
            "sha256": _sha256_bytes(report_bytes),
        },
        "source_data": {
            "path": paths["source_data"],
            "bytes": len(source_bytes),
            "sha256": _sha256_bytes(source_bytes),
        },
        "markdown": {
            "path": paths["markdown"],
            "bytes": len(markdown_bytes),
            "sha256": _sha256_bytes(markdown_bytes),
        },
    }


def write_artifacts(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    active_root = _repo_root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_2_backend_a.json"
        if config_path is None
        else config_path.resolve()
    )
    report = build_report(root=active_root, config_path=active_config)
    source_data = _source_data(report)
    markdown = _markdown(report)
    release_pin = _release_pin(
        root=active_root,
        config_path=active_config,
        report=report,
        source_data=source_data,
        markdown=markdown,
    )
    paths = report["artifact_paths"]
    (active_root / paths["report"]).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (active_root / paths["source_data"]).write_text(
        source_data,
        encoding="utf-8",
        newline="\n",
    )
    (active_root / paths["markdown"]).write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )
    (active_root / paths["release_pin"]).write_text(
        json.dumps(
            release_pin,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def verify_artifacts(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
    expected_analysis_sha256: str | None = None,
) -> dict[str, bool]:
    active_root = _repo_root() if root is None else root.resolve()
    active_config = (
        active_root / "configs/phase9/t9_2_2_backend_a.json"
        if config_path is None
        else config_path.resolve()
    )
    expected = build_report(root=active_root, config_path=active_config)
    paths = expected["artifact_paths"]
    actual = _load_json(active_root / paths["report"])
    source_data = _source_data(expected)
    markdown = _markdown(expected)
    release_pin = _release_pin(
        root=active_root,
        config_path=active_config,
        report=expected,
        source_data=source_data,
        markdown=markdown,
    )
    actual_pin = _load_json(active_root / paths["release_pin"])
    checks = {
        "canonical_report": actual == expected,
        "analysis_sha256": actual.get("analysis_sha256")
        == expected["analysis_sha256"],
        "caller_expected_analysis": (
            expected_analysis_sha256 is None
            or expected["analysis_sha256"] == expected_analysis_sha256
        ),
        "parent_live": all(
            _evaluate_gates(
                expected,
                root=active_root,
                config_path=active_config,
            )[name]
            for name in ("G02_parent_live", "G03_config_live", "G04_source_live")
        ),
        "all_gates": expected["gate_summary"]["all_passed"] is True,
        "all_mutations": expected["mutation_summary"]["all_detected"] is True,
        "source_data": (
            active_root / paths["source_data"]
        ).read_text(encoding="utf-8") == source_data,
        "markdown": (
            active_root / paths["markdown"]
        ).read_text(encoding="utf-8") == markdown,
        "release_pin": actual_pin == release_pin,
        "claims_null": all(
            value is None
            for value in expected["qualification"]["claim_state"].values()
        ),
        "verdict": expected["verdict"]
        == "PASS_T9_2_2_BACKEND_A_QUALIFIED",
    }
    return checks


def _main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-analysis-sha256")
    arguments = parser.parse_args()
    root = _repo_root()
    config = (
        root / "configs/phase9/t9_2_2_backend_a.json"
        if arguments.config is None
        else arguments.config
    )
    if arguments.write:
        report = write_artifacts(root=root, config_path=config)
        print("PASS_T9_2_2_BACKEND_A_ARTIFACTS_WRITTEN")
        print(f"analysis_sha256={report['analysis_sha256']}")
        print(
            f"gates={report['gate_summary']['passed']}/"
            f"{report['gate_summary']['total']}"
        )
        print(
            f"mutations={report['mutation_summary']['detected']}/"
            f"{report['mutation_summary']['total']}"
        )
        return 0 if report["verdict"] == "PASS_T9_2_2_BACKEND_A_QUALIFIED" else 1
    checks = verify_artifacts(
        root=root,
        config_path=config,
        expected_analysis_sha256=arguments.expected_analysis_sha256,
    )
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
