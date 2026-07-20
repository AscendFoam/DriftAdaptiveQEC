from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.8.4"
SCHEMA_VERSION = "t6.8.4-gqf-paper-exact-reproduction-v1"
UPSTREAM_COMMIT = "c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d"
PAPER = ROOT / "relative_papers" / "Non-Markovian_feedback_for_optimized_quantum_error_correction" / "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
INTAKE = ROOT / "docs" / "t6_8_3_gqf_official_intake.json"
RUNNER_MANIFEST = ROOT / "configs" / "gqf_official" / "runner_manifest.json"
PATCH_MANIFEST = ROOT / "configs" / "gqf_official" / "patch_manifest.json"
PROBE_SCRIPT = ROOT / "scripts" / "gqf_paper_reproduction_probe.py"
PROBE_ARTIFACT = ROOT / "docs" / "t6_8_4_gqf_reduced_standard_probe.json"
PREREG = ROOT / "configs" / "gqf_official" / "paper_exact_preregistration.json"
SOURCE_CSV = ROOT / "docs" / "t6_8_4_gqf_reproduction_source_data.csv"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_8_4_gqf_paper_exact_reproduction.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_preregistration(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "t6.8.4-gqf-paper-exact-preregistration-v1",
        "task_id": TASK_ID,
        "materialized_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_sources": {
            "paper_and_supplement": {
                "path": _relative(PAPER),
                "sha256": _sha256(PAPER),
                "doi": "10.1103/PhysRevLett.134.020601",
                "arxiv": "2312.07391",
            },
            "official_repository": "https://github.com/Matteo-Puviani/GQF.git",
            "official_commit": UPSTREAM_COMMIT,
        },
        "paper_protocol": {
            "fock_cutoff": 100,
            "training_Delta": 0.2,
            "evaluation_Delta": {
                "status": "AMBIGUOUS",
                "paper_finite_energy_section": 0.34,
                "best_agent_training_section": 0.2,
            },
            "training_noise": "high",
            "evaluation_noise_levels": ["low", "medium", "high"],
            "dynamics": "Sivak2023 / Fig.S2 timing",
            "full_cycle_us": 10.0,
            "training_full_cycles": 10,
            "training_half_measurements": 20,
            "training_epochs": 1000,
            "agents": 20,
            "batch_sizes_reported": [6, 8],
            "learning_rate": 1.0e-4,
            "controls": 15,
            "rnn_architecture": [10, 256, 256, 15],
            "rnn_cell": "GRU",
            "mf_architecture": [256, 256, 15],
            "training_state": "+Z_L",
            "reward": "final state fidelity",
            "evaluation_full_cycles": 1000,
            "evaluation_logical_states": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
            "reported_sampling": {"NMF": "6x85=510", "standard": "8x64=512"},
            "lifetime_fit": "<P_k>(t)=<P_k>(0) exp(-t/T_k)",
            "channel_lifetime": "3/(1/T_X+1/T_Y+1/T_Z)",
        },
        "published_numeric_anchors": {
            "low_noise_T_Z_standard_cycles": 700.0,
            "low_noise_T_Z_NMF_cycles": 1500.0,
            "low_noise_T_minus_Y_NMF_cycles": 770.0,
            "low_noise_entanglement_infidelity_standard": 1.5e-3,
            "low_noise_entanglement_infidelity_NMF": 4.3e-4,
            "status_of_complete_TX_TY_TZ_Tch_table": "NOT_TABULATED_NUMERICALLY_IN_SOURCE",
        },
        "frozen_acceptance": {
            "categorical_and_integer_configuration": "exact match required",
            "published_two_significant_digit_scalar_relative_tolerance": 0.10,
            "ordering": ["NMF > standard", "NMF > MF", "MF approximately standard"],
            "all_agents_reported": True,
            "all_seed_and_checkpoint_hashes_reported": True,
            "six_state_1000_cycle_raw_trajectories_required": True,
            "absolute_and_ordering_must_both_pass": True,
            "any_source_sufficiency_blocker": "STOP_EXACT_RUN_AND_REPORT_NO_GO",
        },
        "executable_diagnostic_freeze": {
            "probe_script": _relative(PROBE_SCRIPT),
            "probe_script_sha256": _sha256(PROBE_SCRIPT),
            "probe_started_at_utc": probe["started_at_utc"],
            "configuration_enforced_by_script": probe["configuration"],
            "manifest_materialized_after_probe": True,
            "disclosure": "The executable diagnostic configuration was enforced in code before execution; this standalone manifest was materialized afterward. The first run expected 20 steps and failed on the official 21-step behavior; v2 preserves both the 20-step paper prefix and official step 21.",
            "revision_history": [
                {"revision": "probe-v1", "outcome": "REJECTED_756_ROWS_VS_720_EXPECTED", "interpretation": "paper says 20 half-cycles but official max_steps=21 executes 21 env steps"},
                {"revision": "probe-v2", "outcome": "PRESERVE_ALL_21_STEPS", "changed_scientific_thresholds": False},
            ],
        },
        "scope": {
            "full_exact_run_authorized_only_if_source_sufficient": True,
            "reduced_probe_is_exact_evidence": False,
            "project_T2_3_7_may_substitute_official": False,
        },
    }


def _discrepancies() -> list[dict[str, Any]]:
    return [
        {"id": "D01", "field": "official HEAD syntax", "paper": "runnable implementation", "official": "GQF/mesolve.py:13 IndentationError", "blocking": True},
        {"id": "D02", "field": "trained checkpoints", "paper": "best of 20 agents", "official": None, "blocking": True},
        {"id": "D03", "field": "agent seeds", "paper": "20 agents", "official": None, "blocking": True},
        {"id": "D04", "field": "RNN architecture", "paper": [10, 256, 256, 15], "official": [30, 30, 30, 15], "blocking": True},
        {"id": "D05", "field": "MF architecture", "paper": [256, 256, 15], "official": [30, 30, 15], "blocking": True},
        {"id": "D06", "field": "training epochs", "paper": 1000, "official_runner": 101, "official_class_default": 500, "blocking": True},
        {"id": "D07", "field": "Delta", "paper_training": 0.2, "paper_other_section": 0.34, "official_runner": 0.34, "blocking": True},
        {"id": "D08", "field": "bias initialization", "paper_table": 0.01, "paper_text": "[-0.1,0.1]", "official": "[-0.01,0.01] output layer only", "blocking": True},
        {"id": "D09", "field": "model token", "paper": "RNN/GRU", "official_runner": "RNN", "official_class_accepts": "rNN", "blocking": True},
        {"id": "D10", "field": "best-agent saving", "paper": "post-select longest lifetime", "official": "previous_reward initialized to 0 and never updated", "blocking": True},
        {"id": "D11", "field": "20-agent orchestration and selection", "paper": "train/evaluate/post-select 20", "official": None, "blocking": True},
        {"id": "D12", "field": "six-state 1000-cycle evaluator", "paper": "six states and 1000 cycles", "official": "no orchestration or lifetime fitter", "blocking": True},
        {"id": "D13", "field": "raw trajectories", "paper": "510 NMF / 512 standard", "official": None, "blocking": True},
        {"id": "D14", "field": "complete numeric targets", "paper": "figures plus selected scalars", "official": None, "blocking": True},
        {"id": "D15", "field": "fit window/weighting", "paper": "single exponential", "official": "not implemented; fit range and weighting unspecified", "blocking": True},
        {"id": "D16", "field": "max_steps semantics", "paper": "20 measurements / 10 cycles", "official": "max_steps=21 executes 21 env.step calls", "blocking": True},
        {"id": "D17", "field": "GPU execution", "paper": "not a claim", "official_environment": "UNQUALIFIED_CUSOLVER_FATAL on current host", "blocking": True},
        {"id": "D18", "field": "dependency versions", "paper": None, "official": "four unpinned top-level requirements", "blocking": True},
    ]


def _write_source_csv(probe: Mapping[str, Any]) -> None:
    fields = [
        "seed", "state_id", "initial_state", "pauli_axis", "expected_eigenvalue",
        "batch_index", "half_cycle", "full_cycle", "signed_pauli",
        "measurement_probability", "reward", "rho_trace_real", "rho_trace_imag",
    ]
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(probe["rows"])


def evaluate_integrity(report: Mapping[str, Any]) -> dict[str, bool]:
    exact = report["exact_qualification"]
    outcomes = report["paper_exact_outcomes"]
    bindings = report["bindings"]
    return {
        "G01_official_intake_identity_is_reused": report["official_intake"]["commit"] == UPSTREAM_COMMIT and report["official_intake"]["verdict"].startswith("PASS_GQF_OFFICIAL_INTAKE_"),
        "G02_primary_paper_and_official_sources_are_hash_bound": len(report["primary_sources"]["paper_sha256"]) == 64 and len(report["primary_sources"]["official_tree_sha256"]) == 64,
        "G03_preregistration_and_post_probe_revision_are_disclosed": report["preregistration"]["executable_diagnostic_freeze"]["manifest_materialized_after_probe"] is True and report["preregistration"]["executable_diagnostic_freeze"]["revision_history"][0]["outcome"].startswith("REJECTED_") and report["preregistration"]["executable_diagnostic_freeze"]["revision_history"][1]["changed_scientific_thresholds"] is False,
        "G04_blocking_discrepancy_matrix_is_complete": len(report["source_discrepancies"]) >= 18 and all(row["blocking"] for row in report["source_discrepancies"]),
        "G05_missing_fields_are_null_not_guessed": report["guessed_fields"] == [] and len(report["missing_required_fields"]) >= 8,
        "G06_reduced_official_standard_probe_has_frozen_coverage": report["reduced_probe"]["status"] == "PASS_REDUCED_STANDARD_PATH_DIAGNOSTIC" and report["reduced_probe"]["coverage"] == {"rows": 756, "expected_rows": 756, "trajectories": 36, "environment_steps": 378},
        "G07_reduced_probe_physics_checks_and_source_rows_pass": all(report["reduced_probe"]["checks"].values()) and report["source_data_rows"] == 756,
        "G08_unexecuted_exact_metrics_remain_null": all(outcomes[strategy][metric] is None for strategy in ("standard", "MF", "NMF") for metric in ("T_X", "T_Y", "T_Z", "T_ch", "F_avg")),
        "G09_all_twenty_missing_agents_are_explicit": len(report["agent_ledger"]) == 20 and all(row["seed"] is None and row["checkpoint_sha256"] is None and row["status"] == "MISSING_OFFICIAL_AGENT_ARTIFACT" for row in report["agent_ledger"]),
        "G10_exact_no_go_follows_failed_qualification": report["exact_reproduction_status"] == "NO_GO_SOURCE_INCOMPLETE" and exact["passed"] == 0 and exact["failed"] == len(exact["gates"]),
        "G11_downstream_and_claim_boundaries_fail_closed": report["t6_8_5_eligible"] is False and report["claim_boundary"] == {"paper_exact_reproduction": "PROHIBITED", "directional_MF_NMF_ordering": "NOT_ESTABLISHED", "surpass_puviani_nmf": "PROHIBITED", "reduced_official_standard_path": "ESTABLISHED_DIAGNOSTIC_ONLY"},
        "G12_all_live_inputs_and_outputs_are_hash_bound": all(len(item["sha256"]) == 64 for item in bindings.values()),
        "G13_target_specific_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 13,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 13, "detected": 13, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_integrity(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("wrong_commit", "G01_official_intake_identity_is_reused", lambda x: x["official_intake"].update(commit="0" * 40))
    attempt("drop_paper_hash", "G02_primary_paper_and_official_sources_are_hash_bound", lambda x: x["primary_sources"].update(paper_sha256=""))
    attempt("hide_protocol_revision", "G03_preregistration_and_post_probe_revision_are_disclosed", lambda x: x["preregistration"]["executable_diagnostic_freeze"].update(revision_history=[]))
    attempt("drop_blocker", "G04_blocking_discrepancy_matrix_is_complete", lambda x: x.update(source_discrepancies=x["source_discrepancies"][:3]))
    attempt("guess_seed", "G05_missing_fields_are_null_not_guessed", lambda x: x.update(guessed_fields=["agent_seed=0"]))
    attempt("shrink_probe", "G06_reduced_official_standard_probe_has_frozen_coverage", lambda x: x["reduced_probe"].update(coverage={"rows": 1}))
    attempt("hide_trace_failure", "G07_reduced_probe_physics_checks_and_source_rows_pass", lambda x: x["reduced_probe"]["checks"].update(trace_one=False))
    attempt("fabricate_lifetime", "G08_unexecuted_exact_metrics_remain_null", lambda x: x["paper_exact_outcomes"]["NMF"].update(T_Z=1500.0))
    attempt("drop_agents", "G09_all_twenty_missing_agents_are_explicit", lambda x: x.update(agent_ledger=x["agent_ledger"][:1]))
    attempt("promote_exact", "G10_exact_no_go_follows_failed_qualification", lambda x: x.update(exact_reproduction_status="PASS_EXACT"))
    attempt("enable_surpass_lane", "G11_downstream_and_claim_boundaries_fail_closed", lambda x: x.update(t6_8_5_eligible=True))
    attempt("truncate_binding", "G12_all_live_inputs_and_outputs_are_hash_bound", lambda x: x["bindings"]["source_csv"].update(sha256="0"))
    attempt("forge_mutation_count", "G13_target_specific_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 13, "detected": 12, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    intake = _load(INTAKE)
    runner = _load(RUNNER_MANIFEST)
    probe = _load(PROBE_ARTIFACT)
    prereg = _paper_preregistration(probe)
    PREREG.write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_source_csv(probe)

    exact_gates = {
        "E01_unambiguous_physical_and_training_configuration": False,
        "E02_official_twenty_agent_checkpoints_available": False,
        "E03_official_seed_distribution_available": False,
        "E04_paper_network_architectures_match_official_source": False,
        "E05_paper_training_horizon_matches_official_runner": False,
        "E06_official_agent_selection_pipeline_available": False,
        "E07_official_six_state_1000_cycle_evaluator_available": False,
        "E08_official_raw_trajectory_data_available": False,
        "E09_complete_numeric_TX_TY_TZ_Tch_targets_available": False,
        "E10_current_full_GQF_accelerator_path_qualified": False,
        "E11_paper_exact_standard_rerun_completed": False,
        "E12_paper_exact_MF_rerun_completed": False,
        "E13_paper_exact_NMF_rerun_completed": False,
        "E14_absolute_value_tolerances_pass": False,
        "E15_published_ordering_passes": False,
    }
    null_metrics = {"T_X": None, "T_Y": None, "T_Z": None, "T_ch": None, "F_avg": None}
    agent_ledger = [
        {
            "agent_index": index,
            "seed": None,
            "checkpoint_sha256": None,
            "selection_metric": None,
            "T_X": None,
            "T_Y": None,
            "T_Z": None,
            "T_ch": None,
            "F_avg": None,
            "status": "MISSING_OFFICIAL_AGENT_ARTIFACT",
        }
        for index in range(1, 21)
    ]
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_sources": {
            "paper_path": _relative(PAPER),
            "paper_sha256": _sha256(PAPER),
            "paper_doi": "10.1103/PhysRevLett.134.020601",
            "paper_arxiv": "2312.07391",
            "official_repository": runner["repository"],
            "official_commit": runner["commit"],
            "official_tree_sha256": runner["tracked_tree_sha256"],
        },
        "official_intake": {
            "verdict": intake["verdict"],
            "commit": intake["upstream"]["commit"],
            "gpu_status": intake["smoke"]["gpu"]["status"],
        },
        "preregistration": prereg,
        "source_discrepancies": _discrepancies(),
        "missing_required_fields": [
            "20 agent seeds", "20 selected/unselected checkpoints", "selection results",
            "six-state raw 1000-cycle trajectories", "complete TX/TY/TZ/Tch numeric table",
            "fit range and weighting", "paper-matching RNN/MF implementation", "exact dependency versions",
        ],
        "guessed_fields": [],
        "reduced_probe": {
            key: probe[key]
            for key in (
                "scope", "configuration", "coverage", "checks",
                "paper_ten_cycle_prefix_signed_pauli_by_state",
                "official_terminal_signed_pauli_by_state", "elapsed_s", "status",
            )
        },
        "source_data_rows": len(probe["rows"]),
        "paper_exact_outcomes": {
            "standard": null_metrics | {"status": "NOT_RUN_EXACT_PREREQUISITE_FAIL"},
            "MF": null_metrics | {"status": "NOT_RUN_EXACT_PREREQUISITE_FAIL"},
            "NMF": null_metrics | {"status": "NOT_RUN_EXACT_PREREQUISITE_FAIL"},
        },
        "agent_ledger": agent_ledger,
        "exact_qualification": {
            "gates": exact_gates,
            "passed": sum(exact_gates.values()),
            "failed": sum(not value for value in exact_gates.values()),
        },
        "exact_reproduction_status": "NO_GO_SOURCE_INCOMPLETE",
        "t6_8_5_eligible": False,
        "claim_boundary": {
            "paper_exact_reproduction": "PROHIBITED",
            "directional_MF_NMF_ordering": "NOT_ESTABLISHED",
            "surpass_puviani_nmf": "PROHIBITED",
            "reduced_official_standard_path": "ESTABLISHED_DIAGNOSTIC_ONLY",
        },
        "resource_observation": {
            "reduced_probe_elapsed_s": probe["elapsed_s"],
            "reduced_probe_cutoff": 8,
            "paper_cutoff": 100,
            "reduced_probe_training_epochs": 0,
            "paper_training_agent_epochs": 20_000,
            "statement": "No wall-clock extrapolation is claimed; the accelerator fatal and structurally larger paper workload preclude substituting the reduced CPU probe for exact training.",
        },
        "bindings": {
            "implementation": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
            "paper": {"path": _relative(PAPER), "sha256": _sha256(PAPER)},
            "intake": {"path": _relative(INTAKE), "sha256": _sha256(INTAKE)},
            "runner_manifest": {"path": _relative(RUNNER_MANIFEST), "sha256": _sha256(RUNNER_MANIFEST)},
            "patch_manifest": {"path": _relative(PATCH_MANIFEST), "sha256": _sha256(PATCH_MANIFEST)},
            "probe_script": {"path": _relative(PROBE_SCRIPT), "sha256": _sha256(PROBE_SCRIPT)},
            "probe_artifact": {"path": _relative(PROBE_ARTIFACT), "sha256": _sha256(PROBE_ARTIFACT)},
            "preregistration": {"path": _relative(PREREG), "sha256": _sha256(PREREG)},
            "source_csv": {"path": _relative(SOURCE_CSV), "sha256": _sha256(SOURCE_CSV)},
        },
    }
    report["semantic_mutation_audit"] = {"count": 13, "detected": 13, "cases": []}
    report["integrity_gates"] = evaluate_integrity(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["integrity_gates"] = evaluate_integrity(report)
    report["integrity_summary"] = {
        "passed": sum(report["integrity_gates"].values()),
        "failed": sum(not value for value in report["integrity_gates"].values()),
    }
    report["verdict"] = (
        "COMPLETE_GQF_PAPER_EXACT_ATTEMPT_NO_GO_SOURCE_INCOMPLETE"
        if all(report["integrity_gates"].values()) and report["exact_qualification"]["passed"] == 0
        else "FAIL_GQF_PAPER_EXACT_ATTEMPT_INTEGRITY"
    )
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_integrity(report)
    expected = (
        "COMPLETE_GQF_PAPER_EXACT_ATTEMPT_NO_GO_SOURCE_INCOMPLETE"
        if all(gates.values()) and report["exact_qualification"]["passed"] == 0
        else "FAIL_GQF_PAPER_EXACT_ATTEMPT_INTEGRITY"
    )
    if report.get("integrity_gates") != gates or report.get("verdict") != expected or not all(gates.values()):
        raise ValueError("T6.8.4 integrity gates/verdict do not recompute")
    for item in report["bindings"].values():
        path = ROOT / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"T6.8.4 bound artifact drifted: {item['path']}")
    if report["primary_sources"]["official_commit"] != UPSTREAM_COMMIT:
        raise ValueError("T6.8.4 official commit drifted")
    if any(value is not False for value in report["exact_qualification"]["gates"].values()):
        raise ValueError("T6.8.4 exact gate was promoted without evidence")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    report = build_report()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_report(_load(args.artifact))
    print(json.dumps({"verdict": report["verdict"], "integrity": report["integrity_summary"], "exact": report["exact_qualification"], "probe": report["reduced_probe"]["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
