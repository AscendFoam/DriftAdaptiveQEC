"""Create the additive pre-formal T9.2.4 runner/mapping child seal.

The generator is outcome-blind.  It refuses to run after any formal output
exists, preserves the original T9.2.4 preregistration as an immutable parent,
and binds the mapping bridge, runner, independent verifier and tests before a
formal seed may be consumed.
"""

from __future__ import annotations

import argparse
import ast
import copy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TASK_ID = "T9.2.4"
SCHEMA = "PHASE9-DUAL-BACKEND-FORMAL-RUNNER-AMENDMENT-SEAL-V1"
STATUS = "PRE_FORMAL_RUNNER_AND_MAPPING_SEALED"
EXPECTED_PARENT_ANALYSIS = (
    "98e72c457dab941daab270b3bd63eec939564a6f1fedaad061eab59280988695"
)
EXPECTED_PARENT_COMMIT = "4d9d92bbcec211f3c150b70a98f340d67fb975e3"
EXPECTED_TYPED_NULL = (
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "hardware_measured",
    "rank",
)
REQUIRED_ESTIMANDS = {
    "ensemble_density_trace_distance",
    "mean_photon_difference",
    "level_probability_l1",
    "integrated_iq_mean_difference",
    "integrated_iq_covariance_frobenius",
    "iq_two_sample_ks",
    "log_evidence_mean_difference",
    "posterior_mean_l1",
    "logical_trace_decreasing_block",
    "logical_ptm",
    "logical_ptm_entry_difference",
    "logical_survival_difference",
    "reset_success_rate_difference",
    "leakage_residence_rate_difference",
    "short_trajectory_terminal_trace_distance",
    "short_trajectory_observable_mean_difference",
    "conservation_pass_fraction",
    "projector_frobenius",
    "principal_singular",
}
OUTPUT_NAMES = (
    "execution_manifest",
    "report",
    "cell_ledger",
    "source_data",
    "raw_state_archive",
    "markdown",
    "release_pin",
)


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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold an object")
    return value


def _binding(root: Path, relative: str) -> dict[str, object]:
    normalized = relative.replace("\\", "/")
    payload = (root / normalized).read_bytes()
    return {
        "path": normalized,
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


def _parent_tolerances(root: Path) -> dict[str, object]:
    parent = _load(
        root / "configs/phase9/t9_2_4_twin_qualification.json"
    )
    return dict(parent["prefrozen_tolerances"])


def _validate_config(
    root: Path,
    config: Mapping[str, Any],
    *,
    require_no_outputs: bool,
) -> dict[str, bool]:
    if config.get("task_id") != TASK_ID:
        raise ValueError("task id mismatch")
    if (
        config.get("schema_version")
        != "PHASE9-DUAL-BACKEND-FORMAL-RUNNER-AMENDMENT-V1"
    ):
        raise ValueError("amendment schema mismatch")
    if config.get("formal_result_accessed_before_amendment") is not False:
        raise ValueError("formal outcome was accessed before amendment")
    parent_spec = config["parent_preregistration"]
    parent = _load(root / str(parent_spec["path"]))
    if (
        parent_spec["analysis_sha256"] != EXPECTED_PARENT_ANALYSIS
        or parent.get("analysis_sha256") != EXPECTED_PARENT_ANALYSIS
        or parent_spec["git_commit"] != EXPECTED_PARENT_COMMIT
    ):
        raise ValueError("immutable parent lineage mismatch")
    if parent.get("formal_result_accessed") is not False:
        raise ValueError("parent seal no longer pre-formal")

    reason = config["amendment_reason"]
    if (
        reason["formal_seed_or_outcome_used"] is not False
        or reason["threshold_change"] is not False
        or reason["formal_seed_change"] is not False
        or reason["parent_history_rewritten"] is not False
    ):
        raise ValueError("amendment violates outcome-blind additive policy")

    amended_tolerances = dict(config["immutable_parent_tolerances"])
    if amended_tolerances.pop("change_count") != 0:
        raise ValueError("tolerance change count must be zero")
    amended_tolerances.pop("source")
    if amended_tolerances != _parent_tolerances(root):
        raise ValueError("parent tolerance drift")

    bridge = config["logical_bridge"]
    if (
        bridge["bridge_id"]
        != "PHASE9-BACKEND-B-ANALYTIC-MEHLER-FOCK-BRIDGE-V1"
        or bridge["projector_delta"] != 0.34
        or bridge["tail_tolerance"] != 1e-12
        or "symmetric parity-constrained" not in bridge["support_rule"]
        or "tanh(projector_delta^2)" not in bridge["amplitude_variance"]
        or "sech(projector_delta^2)" not in bridge["center_contraction"]
    ):
        raise ValueError("logical bridge physical convention drift")
    forbidden = set(bridge["forbidden_runtime_imports"])
    if forbidden != {
        "physics.phase9_backend_a",
        "physics.finite_energy_gkp",
        "physics.fock_density_model",
        "physics.fock_sbs_cycle",
    }:
        raise ValueError("logical bridge forbidden-import ledger drift")

    pilot = config["deterministic_mapping_pilot"]
    if pilot["formal_seed_or_outcome_accessed"] is not False:
        raise ValueError("mapping pilot accessed formal outcome")
    if pilot["cutoffs"] != [8, 12, 16]:
        raise ValueError("mapping pilot cutoff ledger drift")
    if not (
        float(pilot["released_native_projector_frobenius"]["12"]) > 0.30
        and max(
            float(value)
            for value in pilot[
                "analytic_bridge_projector_frobenius"
            ].values()
        )
        < 1e-9
        and min(
            float(value)
            for value in pilot[
                "analytic_bridge_minimum_principal_singular"
            ].values()
        )
        >= 0.95
    ):
        raise ValueError("mapping negative/repair witness invalid")
    if (
        float(pilot["captured_probability"]["8"][0]) >= 0.90
        or "does not prove physical cutoff convergence"
        not in pilot["disclosure"]
    ):
        raise ValueError("fixed-cutoff/cutoff-convergence disclosure missing")

    initial = config["initial_state_definitions"]
    if set(initial) != {
        "vacuum_g",
        "one_g",
        "zero_one_plus_g",
        "vacuum_e",
        "vacuum_f",
        "initial_drift",
        "initial_leakage_age",
        "initial_round_index",
    }:
        raise ValueError("initial-state schema drift")
    if (
        initial["initial_drift"] != [0.0] * 5
        or initial["initial_leakage_age"] != 0
        or initial["initial_round_index"] != 0
    ):
        raise ValueError("initial causal state drift")

    action = config["action_and_fault_semantics"]
    if action["one_round_actions"] != [
        "IDLE",
        "X",
        "Z",
        "XZ",
        "RESET",
        "HOLD",
        "LKG_HOLD",
    ]:
        raise ValueError("action ledger drift")
    if set(action["fault_action_sequences"]) != {
        "step",
        "telegraph",
        "burst",
        "compound",
    }:
        raise ValueError("fault action-sequence ledger drift")
    if "never observe scenario identity" not in action["intervention_rule"]:
        raise ValueError("fault intervention leaks hidden state")

    convergence = config["cutoff_convergence_submatrix"]
    if (
        convergence["cutoffs"] != [8, 12]
        or convergence["shared_states"]
        != [
            "vacuum_g",
            "one_g",
            "zero_one_plus_g",
            "vacuum_e",
            "vacuum_f",
        ]
        or convergence["logical_states"]
        != ["0", "1", "+", "-", "+i", "-i"]
        or convergence["actions"] != ["IDLE", "XZ", "RESET"]
        or convergence["fault_scenarios"]
        != ["step", "telegraph", "burst", "compound"]
        or convergence["samples_per_state_action_backend_cutoff"] != 16
        or convergence["trajectories_per_fault_backend_cutoff"] != 8
        or convergence["horizon"] != 12
        or convergence["additional_cutoff_12_backend_rounds"] != 1824
        or convergence[
            "total_unique_formal_backend_rounds_after_amendment"
        ]
        != 16800
    ):
        raise ValueError("common cutoff confirmation matrix drift")

    seed = config["seed_expansion"]
    if (
        seed["cross_backend_pairing"] is not False
        or seed["cross_cutoff_pairing_within_backend"] is not True
        or seed["mutable_rng_consumption_forbidden"] is not True
        or "first 16" not in seed["cutoff_convergence_cells"]
        or "first 8" not in seed["cutoff_convergence_cells"]
    ):
        raise ValueError("seed expansion/cluster contract drift")

    if set(config["estimands"]) != REQUIRED_ESTIMANDS:
        raise ValueError("estimand schema drift")
    if (
        "one half" not in config["estimands"][
            "ensemble_density_trace_distance"
        ]
        or "no projection-conditioned renormalization"
        not in config["estimands"]["logical_trace_decreasing_block"]
        or "unconditional" not in config["estimands"]["logical_ptm"]
    ):
        raise ValueError("estimand definition permits metric switching")

    statistics = config["statistical_procedure"]
    if (
        statistics["confidence"] != 0.95
        or statistics["bootstrap_resamples"] != 2000
        or statistics["bootstrap_seed"] != 960000
        or statistics["standard_error_floor"] != 1e-12
        or "seed_position cluster" not in statistics["resampling_unit"]
        or "independently" not in statistics["backend_sampling"]
        or "same within-backend" not in statistics["cross_cutoff_dependence"]
        or "maximum" not in statistics["max_t_statistic"]
        or "method=higher" not in statistics["critical_value"]
        or statistics["mean_only_rescue"] is not False
        or statistics["cell_deletion"] is not False
    ):
        raise ValueError("statistical procedure drift")

    policy = config["failure_and_release_policy"]
    if (
        policy["pass_verdict"] != "PASS_T9_2_4_DUAL_BACKEND_QUALIFIED"
        or policy["scientific_fail_verdict"] != "NO_GO_TWIN_QUALIFICATION"
        or policy["infrastructure_fail_verdict"] != "INCOMPLETE_FAIL_CLOSED"
        or policy["early_exit_on_first_metric_failure"] is not False
        or policy["released_after_scientific_fail"] != ["T9.2.6"]
        or "T9.2.5" not in policy["blocked_after_scientific_fail"]
        or policy["no_formal_lifetime_on_fail"] is not True
        or policy["no_retuning_after_formal_access"] is not True
    ):
        raise ValueError("failure/release policy drift")
    claim = config["claim_boundary"]
    if (
        claim["qualified_if_pass"]
        != ["dual_backend_agreement_for_prefrozen_synthetic_task"]
        or tuple(claim["typed_null"]) != EXPECTED_TYPED_NULL
        or "official Puviani reproduction or surpass"
        not in claim["forbidden_claim_expansions"]
    ):
        raise ValueError("claim boundary drift")

    artifacts = config["artifact_paths"]
    required_artifacts = {
        "amendment_seal",
        "amendment_generator",
        "amendment_tests",
        "formal_runner",
        "independent_verifier",
        "bridge_tests",
        "runner_tests",
        "verifier_tests",
        "execution_manifest",
        "report",
        "cell_ledger",
        "source_data",
        "raw_state_archive",
        "markdown",
        "release_pin",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError("artifact path schema drift")
    if require_no_outputs:
        existing = [
            name
            for name in OUTPUT_NAMES
            if (root / artifacts[name]).exists()
        ]
        if existing:
            raise ValueError(
                "formal outputs already exist before child seal: "
                + ",".join(existing)
            )

    bridge_source = (
        root / str(config["logical_bridge"]["implementation"])
    )
    bridge_imports = _imports(bridge_source)
    if any(
        token in module
        for module in bridge_imports
        for token in (
            "phase9_backend_a",
            "finite_energy_gkp",
            "fock_density_model",
            "fock_sbs_cycle",
        )
    ):
        raise ValueError("bridge source imports a forbidden projector/runtime")
    bridge_text = bridge_source.read_text(encoding="utf-8")
    if "expm(" in bridge_text or "_symmetric_parity_indices" not in bridge_text:
        raise ValueError("bridge restored truncated expm or lost symmetric support")

    runner_path = root / artifacts["formal_runner"]
    runner_text = runner_path.read_text(encoding="utf-8")
    runner_imports = _imports(runner_path)
    if not any("phase9_backend_a" in value for value in runner_imports):
        raise ValueError("runner no longer executes backend A")
    if not any("phase9_backend_b" in value for value in runner_imports):
        raise ValueError("runner no longer executes backend B")
    if (
        "--seed" in runner_text
        or "--repeats" in runner_text
        or "--tolerance" in runner_text
        or "survival * np.asarray" not in runner_text
        or "expected_rows" not in runner_text
    ):
        raise ValueError("runner permits formal override or lost no-postselection")

    verifier_path = root / artifacts["independent_verifier"]
    verifier_text = verifier_path.read_text(encoding="utf-8")
    if any(value.startswith("physics") for value in _imports(verifier_path)):
        raise ValueError("independent verifier imports physics runtime")
    if (
        "bootstrap_resamples" not in verifier_text
        or "method=\"higher\"" not in verifier_text
        or "INCOMPLETE_FAIL_CLOSED" not in verifier_text
        or "TYPED_NULL_FIELDS" not in verifier_text
    ):
        raise ValueError("verifier lost frozen statistics/failure/claim logic")

    return {
        "G01_parent_preregistration_immutable": True,
        "G02_formal_outcome_unaccessed": True,
        "G03_additive_not_rewrite": True,
        "G04_parent_tolerances_unchanged": True,
        "G05_native_negative_witness_preserved": True,
        "G06_analytic_mapping_witness_passes": True,
        "G07_cutoff_nonconvergence_disclosed": True,
        "G08_bridge_physical_identity_frozen": True,
        "G09_bridge_independence_static": True,
        "G10_symmetric_tail_support_frozen": True,
        "G11_initial_states_exact": True,
        "G12_action_sequences_exact": True,
        "G13_fault_interventions_causal": True,
        "G14_seed_expansion_exact": True,
        "G15_backend_seed_independence": True,
        "G16_cross_cutoff_pairing_frozen": True,
        "G17_cutoff_confirmation_not_demo": True,
        "G18_total_row_accounting_exact": True,
        "G19_all_estimands_operational": True,
        "G20_trace_distance_factor_frozen": True,
        "G21_no_postselection_logical_ptm": True,
        "G22_cluster_bootstrap_frozen": True,
        "G23_max_t_familywise_frozen": True,
        "G24_no_mean_or_cell_rescue": True,
        "G25_three_way_verdict_frozen": True,
        "G26_failure_propagation_complete": True,
        "G27_claim_boundary_typed_null": True,
        "G28_runner_live_structure": True,
        "G29_independent_verifier_live_structure": True,
        "G30_formal_outputs_absent": True,
    }


def _set(value: dict[str, Any], path: Sequence[str], replacement: object) -> None:
    cursor: dict[str, Any] = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


def _mutation_audit(
    root: Path,
    config: Mapping[str, Any],
) -> list[dict[str, object]]:
    mutations: tuple[
        tuple[str, str, Callable[[dict[str, Any]], None]],
        ...,
    ] = (
        ("M01", "G01_parent_preregistration_immutable", lambda c: _set(c, ("parent_preregistration", "analysis_sha256"), "0" * 64)),
        ("M02", "G02_formal_outcome_unaccessed", lambda c: _set(c, ("formal_result_accessed_before_amendment",), True)),
        ("M03", "G03_additive_not_rewrite", lambda c: _set(c, ("amendment_reason", "parent_history_rewritten"), True)),
        ("M04", "G04_parent_tolerances_unchanged", lambda c: _set(c, ("immutable_parent_tolerances", "maximum_logical_ptm_entry_difference"), 0.13)),
        ("M05", "G05_native_negative_witness_preserved", lambda c: _set(c, ("deterministic_mapping_pilot", "released_native_projector_frobenius", "12"), 0.20)),
        ("M06", "G06_analytic_mapping_witness_passes", lambda c: _set(c, ("deterministic_mapping_pilot", "analytic_bridge_projector_frobenius", "12"), 0.31)),
        ("M07", "G07_cutoff_nonconvergence_disclosed", lambda c: _set(c, ("deterministic_mapping_pilot", "disclosure"), "converged")),
        ("M08", "G08_bridge_physical_identity_frozen", lambda c: _set(c, ("logical_bridge", "center_contraction"), "1")),
        ("M09", "G09_bridge_independence_static", lambda c: _set(c, ("logical_bridge", "forbidden_runtime_imports"), [])),
        ("M10", "G10_symmetric_tail_support_frozen", lambda c: _set(c, ("logical_bridge", "support_rule"), "range(-h,h+1)")),
        ("M11", "G11_initial_states_exact", lambda c: _set(c, ("initial_state_definitions", "initial_drift"), [0.1] * 5)),
        ("M12", "G12_action_sequences_exact", lambda c: _set(c, ("action_and_fault_semantics", "one_round_actions"), ["IDLE"])),
        ("M13", "G13_fault_interventions_causal", lambda c: _set(c, ("action_and_fault_semantics", "intervention_rule"), "controller observes scenario")),
        ("M14", "G14_seed_expansion_exact", lambda c: _set(c, ("seed_expansion", "cutoff_convergence_cells"), "arbitrary")),
        ("M15", "G15_backend_seed_independence", lambda c: _set(c, ("seed_expansion", "cross_backend_pairing"), True)),
        ("M16", "G16_cross_cutoff_pairing_frozen", lambda c: _set(c, ("seed_expansion", "cross_cutoff_pairing_within_backend"), False)),
        ("M17", "G17_cutoff_confirmation_not_demo", lambda c: _set(c, ("cutoff_convergence_submatrix", "samples_per_state_action_backend_cutoff"), 2)),
        ("M18", "G18_total_row_accounting_exact", lambda c: _set(c, ("cutoff_convergence_submatrix", "total_unique_formal_backend_rounds_after_amendment"), 14976)),
        ("M19", "G19_all_estimands_operational", lambda c: c["estimands"].pop("iq_two_sample_ks")),
        ("M20", "G20_trace_distance_factor_frozen", lambda c: _set(c, ("estimands", "ensemble_density_trace_distance"), "nuclear norm")),
        ("M21", "G21_no_postselection_logical_ptm", lambda c: _set(c, ("estimands", "logical_trace_decreasing_block"), "conditional density")),
        ("M22", "G22_cluster_bootstrap_frozen", lambda c: _set(c, ("statistical_procedure", "resampling_unit"), "round")),
        ("M23", "G23_max_t_familywise_frozen", lambda c: _set(c, ("statistical_procedure", "bootstrap_resamples"), 100)),
        ("M24", "G24_no_mean_or_cell_rescue", lambda c: _set(c, ("statistical_procedure", "cell_deletion"), True)),
        ("M25", "G25_three_way_verdict_frozen", lambda c: _set(c, ("failure_and_release_policy", "infrastructure_fail_verdict"), "NO_GO_TWIN_QUALIFICATION")),
        ("M26", "G26_failure_propagation_complete", lambda c: _set(c, ("failure_and_release_policy", "released_after_scientific_fail"), ["T9.2.5", "T9.2.6"])),
        ("M27", "G27_claim_boundary_typed_null", lambda c: _set(c, ("claim_boundary", "typed_null"), ["rank"])),
        ("M28", "G28_runner_live_structure", lambda c: _set(c, ("artifact_paths", "formal_runner"), "physics/phase9_backend_a.py")),
        ("M29", "G29_independent_verifier_live_structure", lambda c: _set(c, ("artifact_paths", "independent_verifier"), "physics/phase9_backend_a.py")),
        ("M30", "G30_formal_outputs_absent", lambda c: _set(c, ("artifact_paths", "report"), "configs/phase9/t9_2_4_twin_qualification.json")),
    )
    results: list[dict[str, object]] = []
    for mutation_id, target, mutate in mutations:
        candidate = copy.deepcopy(config)
        mutate(candidate)
        detected = False
        try:
            _validate_config(root, candidate, require_no_outputs=True)
        except (KeyError, TypeError, ValueError):
            detected = True
        results.append(
            {
                "mutation_id": mutation_id,
                "target_gate": target,
                "detected": detected,
            }
        )
    return results


def build_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    gates = _validate_config(root, config, require_no_outputs=True)
    mutations = _mutation_audit(root, config)
    if not all(gates.values()):
        raise RuntimeError("pre-formal amendment gate failed")
    if not all(item["detected"] is True for item in mutations):
        missed = [
            item["mutation_id"]
            for item in mutations
            if item["detected"] is not True
        ]
        raise RuntimeError("semantic mutation survived: " + ",".join(missed))
    artifacts = config["artifact_paths"]
    live_bindings = {
        "parent_preregistration": _binding(
            root,
            config["parent_preregistration"]["path"],
        ),
        "parent_config": _binding(
            root,
            "configs/phase9/t9_2_4_twin_qualification.json",
        ),
        "amendment_config": _binding(
            root,
            "configs/phase9/t9_2_4_formal_runner_amendment.json",
        ),
        "amendment_generator": _binding(
            root,
            artifacts["amendment_generator"],
        ),
        "mapping_bridge": _binding(
            root,
            config["logical_bridge"]["implementation"],
        ),
        "formal_runner": _binding(root, artifacts["formal_runner"]),
        "independent_verifier": _binding(
            root,
            artifacts["independent_verifier"],
        ),
        "bridge_tests": _binding(root, artifacts["bridge_tests"]),
        "runner_tests": _binding(root, artifacts["runner_tests"]),
        "verifier_tests": _binding(root, artifacts["verifier_tests"]),
        "amendment_tests": _binding(root, artifacts["amendment_tests"]),
        "lfs_rules": _binding(root, ".gitattributes"),
    }
    seal: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "amendment_id": config["amendment_id"],
        "status": STATUS,
        "formal_result_accessed": False,
        "parent_preregistration_analysis_sha256": EXPECTED_PARENT_ANALYSIS,
        "parent_git_commit": EXPECTED_PARENT_COMMIT,
        "amendment_config": live_bindings["amendment_config"],
        "live_bindings": live_bindings,
        "formal_row_accounting": {
            "parent_backend_rounds": 14976,
            "additional_cutoff_12_backend_rounds": 1824,
            "total_unique_backend_rounds": 16800,
        },
        "mapping_resolution": {
            "native_cutoff_12_projector_frobenius": config[
                "deterministic_mapping_pilot"
            ]["released_native_projector_frobenius"]["12"],
            "bridge_cutoff_12_projector_frobenius": config[
                "deterministic_mapping_pilot"
            ]["analytic_bridge_projector_frobenius"]["12"],
            "tolerance_unchanged": True,
            "physical_cutoff_convergence_claim": None,
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "all_passed": all(gates.values()),
        },
        "semantic_mutation_audit": mutations,
        "mutation_summary": {
            "detected": sum(
                item["detected"] is True for item in mutations
            ),
            "total": len(mutations),
            "all_detected": all(
                item["detected"] is True for item in mutations
            ),
        },
        "claim_state": {field: None for field in EXPECTED_TYPED_NULL},
        "all_gates_passed": True,
        "formal_command": (
            "python -m cnn_fpga.benchmark.phase9_dual_backend_qualification "
            "&& python -m cnn_fpga.benchmark.phase9_dual_backend_verifier"
        ),
    }
    seal["analysis_sha256"] = _sha(
        _canonical(seal).encode("utf-8")
    )
    return seal


def write_seal(root: Path, seal: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_root())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase9/t9_2_4_formal_runner_amendment.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/t9_2_4_formal_runner_amendment_seal.json"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    output_path = (
        args.output
        if args.output.is_absolute()
        else root / args.output
    )
    config = _load(config_path)
    seal = build_seal(root, config)
    write_seal(root, seal, output_path)
    print(
        _canonical(
            {
                "task_id": TASK_ID,
                "analysis_sha256": seal["analysis_sha256"],
                "gates": seal["gate_summary"],
                "mutations": seal["mutation_summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_PARENT_ANALYSIS",
    "EXPECTED_TYPED_NULL",
    "OUTPUT_NAMES",
    "SCHEMA",
    "STATUS",
    "build_seal",
    "write_seal",
]
