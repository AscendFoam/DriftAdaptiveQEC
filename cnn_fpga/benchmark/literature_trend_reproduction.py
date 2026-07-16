"""Machine-readable literature-trend reproduction registry for T5.0.1.

The registry deliberately separates three things that are easy to conflate:
external reference facts, already-qualified project evidence, and future
holdout gates.  Building this table does not execute a new physics simulation
and a pending/reference row can never be promoted to a reproduced result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "T5.0.1"
SCHEMA_VERSION = 1
PROTOCOL_ID = "T501-LITERATURE-TREND-REPRODUCTION-REGISTRY-V1"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t5_0_1_literature_trend_reproduction.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_0_1_literature_trend_reproduction_source_data.csv")

ARTIFACT_BINDINGS = {
    "T2.0.5": Path("docs/t2_0_5_displacement_fault_trend.json"),
    "T2.0.6": Path("docs/t2_0_6_occupancy_correlation.json"),
    "T2.2.2": Path("docs/t2_2_2_protocol_ancilla_validation.json"),
    "T2.3.3": Path("docs/t2_3_3_cross_fidelity_validation.json"),
    "T2.3.7": Path("docs/t2_3_7_nmf_directional_ranking.json"),
    "T4.4.5": Path("docs/t4_4_5_teacher_student_branch_freeze.json"),
}

CAMPAGNE_PATH = Path(
    "relative_papers/Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator/"
    "Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator.md"
)
SIVAK_PATH = Path(
    "relative_papers/Real-time_quantum_error_correction_beyond_break-even/"
    "Real-time_quantum_error_correction_beyond_break-even.md"
)
PUVIANI_PATH = Path(
    "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
    "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
)

SOURCES = (
    {
        "source_id": "SRC-CAMPAGNE-2020",
        "title": "Quantum error correction of a qubit encoded in grid states of an oscillator",
        "year": 2020,
        "publication": "Nature 584, 368-372",
        "identifier": "doi:10.1038/s41586-020-2603-3",
        "official_url": "https://www.nature.com/articles/s41586-020-2603-3",
        "source_class": "peer_reviewed_experiment",
        "verified_on": "2026-07-16",
    },
    {
        "source_id": "SRC-SIVAK-2023",
        "title": "Real-time quantum error correction beyond break-even",
        "year": 2023,
        "publication": "Nature 616, 50-55",
        "identifier": "doi:10.1038/s41586-023-05782-6",
        "official_url": "https://www.nature.com/articles/s41586-023-05782-6",
        "source_class": "peer_reviewed_experiment",
        "verified_on": "2026-07-16",
    },
    {
        "source_id": "SRC-PUVIANI-PRL-2025",
        "title": "Non-Markovian Feedback for Optimized Quantum Error Correction",
        "year": 2025,
        "publication": "Physical Review Letters 134, 020601",
        "identifier": "doi:10.1103/PhysRevLett.134.020601",
        "official_url": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020601",
        "source_class": "peer_reviewed_theory_simulation",
        "verified_on": "2026-07-16",
    },
    {
        "source_id": "SRC-RALPH-2024",
        "title": "Noise Transfer Approach to GKP Quantum Circuits",
        "year": 2024,
        "publication": "Entropy 26, 874",
        "identifier": "doi:10.3390/e26100874",
        "official_url": "https://www.mdpi.com/1099-4300/26/10/874",
        "source_class": "peer_reviewed_theory",
        "verified_on": "2026-07-16",
    },
    {
        "source_id": "SRC-MARQVERSEN-2025",
        "title": "Performance analysis of GKP error correction",
        "year": 2025,
        "publication": "arXiv preprint",
        "identifier": "arXiv:2505.14775v1",
        "official_url": "https://arxiv.org/abs/2505.14775",
        "source_class": "primary_preprint_theory",
        "verified_on": "2026-07-16",
    },
    {
        "source_id": "SRC-CHEN-2026",
        "title": "Optimized Gottesman-Kitaev-Preskill Error Correction via Tunable Preprocessing",
        "year": 2026,
        "publication": "arXiv preprint",
        "identifier": "arXiv:2604.08247v1",
        "official_url": "https://arxiv.org/abs/2604.08247",
        "source_class": "primary_preprint_theory",
        "verified_on": "2026-07-16",
    },
    {
        "source_id": "SRC-FONTBOTE-2026",
        "title": "Error Correction of Beamsplitter-Generated Entangled GKP States",
        "year": 2026,
        "publication": "arXiv preprint",
        "identifier": "arXiv:2605.08009v1",
        "official_url": "https://arxiv.org/abs/2605.08009",
        "source_class": "primary_preprint_experiment",
        "verified_on": "2026-07-16",
    },
)

LOCAL_SOURCE_ANCHORS = (
    {
        "anchor_id": "A-T501-CAMPAGNE-STRUCTURE",
        "source_id": "SRC-CAMPAGNE-2020",
        "path": CAMPAGNE_PATH,
        "line": 35,
        "fragment": "alternates indefinitely two peak-sharpening rounds and two envelope-trimming rounds",
    },
    {
        "anchor_id": "A-T501-CAMPAGNE-LIFETIME",
        "source_id": "SRC-CAMPAGNE-2020",
        "path": CAMPAGNE_PATH,
        "line": 77,
        "fragment": "extends the lifetime of the 3 Bloch vector components",
    },
    {
        "anchor_id": "A-T501-SIVAK-CYCLE",
        "source_id": "SRC-SIVAK-2023",
        "path": SIVAK_PATH,
        "line": 45,
        "fragment": "duration of a QEC cycle is $ t_{c}=2\\times 4.924\\mu s $",
    },
    {
        "anchor_id": "A-T501-SIVAK-DISPLACEMENT",
        "source_id": "SRC-SIVAK-2023",
        "path": SIVAK_PATH,
        "line": 115,
        "fragment": "a displacement of amplitude $ l_{S}/4 $ makes a large-distance error",
    },
    {
        "anchor_id": "A-T501-SIVAK-OCCUPANCY",
        "source_id": "SRC-SIVAK-2023",
        "path": SIVAK_PATH,
        "line": 181,
        "fragment": "code projector $ \\langle \\Pi_{0}\\rangle=0.825\\pm 0.003 $",
    },
    {
        "anchor_id": "A-T501-PUVIANI-DIRECTION",
        "source_id": "SRC-PUVIANI-PRL-2025",
        "path": PUVIANI_PATH,
        "line": 49,
        "fragment": "increasing the logical qubit's lifetime significantly",
    },
    {
        "anchor_id": "A-T501-PUVIANI-HORIZON",
        "source_id": "SRC-PUVIANI-PRL-2025",
        "path": PUVIANI_PATH,
        "line": 95,
        "fragment": "training is performed on a fixed initial logical state",
    },
    {
        "anchor_id": "A-T501-PUVIANI-LIFETIME",
        "source_id": "SRC-PUVIANI-PRL-2025",
        "path": PUVIANI_PATH,
        "line": 99,
        "fragment": "increases from $ T_{Z} $ (std.",
    },
)

STATUS_VALUES = {
    "QUALIFIED_DIRECTIONAL_PASS",
    "STRUCTURE_IMPLEMENTED_NOT_NUMERIC_REPRODUCTION",
    "REGISTERED_PENDING",
    "REFERENCE_ONLY",
    "NEGATIVE_BOUNDARY_VERIFIED",
    "REPORTING_TEMPLATE_ONLY",
}


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_bound_artifacts(
    paths: Mapping[str, str | Path] = ARTIFACT_BINDINGS,
) -> dict[str, dict[str, Any]]:
    return {
        task_id: json.loads(_repo_path(path).read_text(encoding="utf-8"))
        for task_id, path in paths.items()
    }


def _artifact_machine_pass(task_id: str, payload: Mapping[str, Any]) -> bool:
    if task_id in {"T2.0.5", "T2.0.6"}:
        gate = payload.get("gate", {})
        checks = gate.get("checks", ()) if isinstance(gate, Mapping) else ()
        return bool(gate.get("passed") is True and checks and all(row.get("passed") is True for row in checks))
    if task_id == "T2.2.2":
        checks = payload.get("checks", {})
        secondary = payload.get("secondary_protocols", ())
        return bool(
            isinstance(checks, Mapping)
            and checks
            and all(value is True for value in checks.values())
            and secondary
            and all(row.get("executable") is False for row in secondary)
        )
    if task_id == "T2.3.3":
        checks = payload.get("checks", {})
        return bool(
            payload.get("passed") is True
            and isinstance(checks, Mapping)
            and checks
            and all(value is True for value in checks.values())
        )
    if task_id in {"T2.3.7", "T4.4.5"}:
        gates = payload.get("gates", {})
        return bool(
            payload.get("status") == "PASS"
            and isinstance(gates, Mapping)
            and gates
            and all(value is True for value in gates.values())
        )
    raise KeyError(task_id)


def inspect_local_anchors() -> list[dict[str, Any]]:
    inspected: list[dict[str, Any]] = []
    for anchor in LOCAL_SOURCE_ANCHORS:
        path = _repo_path(anchor["path"])
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        line = int(anchor["line"])
        actual = lines[line - 1] if 1 <= line <= len(lines) else None
        inspected.append(
            {
                **{key: (str(value) if isinstance(value, Path) else value) for key, value in anchor.items()},
                "actual_line_sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest() if actual else None,
                "passed": bool(actual is not None and anchor["fragment"] in actual),
            }
        )
    return inspected


def _target(
    target_id: str,
    topic: str,
    source_ids: Sequence[str],
    hierarchy_role: str,
    target_type: str,
    reference_target: Mapping[str, Any],
    tolerance_rule: Mapping[str, Any],
    calibration_or_holdout_use: str,
    model_selection_access: bool,
    current_status: str,
    current_observation: Mapping[str, Any],
    evidence_artifacts: Sequence[str],
    next_gate: str,
    prohibited_transfer: Sequence[str],
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "topic": topic,
        "source_ids": list(source_ids),
        "hierarchy_role": hierarchy_role,
        "target_type": target_type,
        "reference_target": dict(reference_target),
        "tolerance_rule": dict(tolerance_rule),
        "calibration_or_holdout_use": calibration_or_holdout_use,
        "model_selection_access": model_selection_access,
        "current_status": current_status,
        "current_observation": dict(current_observation),
        "evidence_artifacts": list(evidence_artifacts),
        "next_gate": next_gate,
        "prohibited_transfer": list(prohibited_transfer),
    }


def build_targets(artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    t205 = artifacts["T2.0.5"]
    t206 = artifacts["T2.0.6"]
    t222 = artifacts["T2.2.2"]
    t233 = artifacts["T2.3.3"]
    t237 = artifacts["T2.3.7"]
    t445 = artifacts["T4.4.5"]
    displacement_checks = {row["check_id"]: row for row in t205["gate"]["checks"]}
    occupancy_checks = {row["check_id"]: row for row in t206["gate"]["checks"]}
    low_point = next(point for point in t233["points"] if float(point["squeezing_db"]) == 3.0)
    paired = t237["paired_bootstrap"]["nmf_minus_mf_logical_z_lifetime"]
    counter = t445["counterevidence"]

    return [
        _target(
            "LT-2020-STRUCTURE",
            "Campagne 2020 four-round sharpen--trim protocol structure",
            ["SRC-CAMPAGNE-2020"],
            "primary_cross_validation",
            "STRUCTURAL_DIRECTIONAL",
            {"peak_sharpen_rounds": 2, "envelope_trim_rounds": 2, "repeat": "indefinite"},
            {"kind": "exact_protocol_structure", "round_order_must_remain_native": True},
            "calibration_only",
            True,
            "STRUCTURE_IMPLEMENTED_NOT_NUMERIC_REPRODUCTION",
            {
                "secondary_protocols_non_executable": t222["checks"]["secondary_protocols_remain_non_executable"],
                "sharpen_trim_fault_overlay_checks_pass": all(t222["checks"].values()),
            },
            [str(ARTIFACT_BINDINGS["T2.2.2"])],
            "T5.0.2 independent protocol-trend holdout",
            ["do not merge sharpen--trim observations or timing into sBs", "not a master-equation numeric reproduction"],
        ),
        _target(
            "LT-2020-QEC-ON-OFF",
            "Campagne 2020 Pauli lifetime QEC-on versus QEC-off",
            ["SRC-CAMPAGNE-2020"],
            "primary_cross_validation",
            "NUMERIC_REFERENCE_AND_DIRECTION",
            {"direction": "QEC_on_lifetime_above_QEC_off_for_X_Y_Z", "qec_on_us": {"X": 275, "Y": 160, "Z": 275}},
            {"kind": "future_project_holdout", "criterion": "paired_95pct_CI_for_on_minus_off_above_zero_for_each_Pauli", "match_external_microseconds": False},
            "future_holdout_preregistered",
            False,
            "REFERENCE_ONLY",
            {"project_numeric_reproduction_executed": False},
            [],
            "T5.3.1 logical-channel QEC-on/off reconstruction",
            ["external microseconds are not a project acceptance target", "do not transfer superconducting timing to FPGA or simulator timing"],
        ),
        _target(
            "LT-2023-DISPLACEMENT",
            "Sivak 2023 large-distance displacement recovery trend",
            ["SRC-SIVAK-2023"],
            "main_digital_twin",
            "DIRECTIONAL",
            {"peak_recovery_depth_at_normalized_displacement": 0.25, "mirror_about": 0.25},
            {"kind": "registered_effective_model", "peak_absolute_tolerance": 0.0625, "branch_spearman_abs_min": 0.95, "recovered_fraction_min": 0.98},
            "calibration_only",
            True,
            "QUALIFIED_DIRECTIONAL_PASS",
            {
                "peak_location": displacement_checks["peak_near_lS_over_4"]["observed"],
                "left_spearman": displacement_checks["left_branch_monotone_increasing"]["observed"],
                "right_spearman": displacement_checks["right_branch_monotone_decreasing"]["observed"],
                "recovered_fraction": displacement_checks["midpoint_recovers_within_horizon"]["observed"],
            },
            [str(ARTIFACT_BINDINGS["T2.0.5"])],
            "T5.2.1 independent displacement injection",
            ["coarse error-space trend is not a Fock/device reproduction", "no experimental recovery-time equality claim"],
        ),
        _target(
            "LT-2023-OCCUPANCY-CORRELATION",
            "Sivak 2023 code occupancy and leakage-correlation-tail trend",
            ["SRC-SIVAK-2023"],
            "main_digital_twin",
            "NUMERIC_AND_DIRECTIONAL",
            {"code_projector_occupancy": 0.825, "long_leakage_removal_reduces_tail": True},
            {"kind": "registered_effective_model", "occupancy_absolute_tolerance": 0.02, "paired_tail_CI_low_min": 0.0003, "post_removal_tail_abs_max": 0.0015},
            "calibration_only",
            True,
            "QUALIFIED_DIRECTIONAL_PASS",
            {
                "occupancy_absolute_error": occupancy_checks["hidden_occupancy_near_reference"]["observed"],
                "tail_paired_ci_low": occupancy_checks["tail_shrink_paired_ci_positive"]["observed"],
                "post_removal_tail_abs": occupancy_checks["post_removal_tail_small"]["observed"],
            },
            [str(ARTIFACT_BINDINGS["T2.0.6"])],
            "T5.2.3 multilevel leakage/reset holdout",
            ["effective hidden occupancy is not reconstructed experimental density", "post-selection is not an online leakage controller"],
        ),
        _target(
            "LT-2023-GAIN-TIMING",
            "Sivak 2023 gain and native cycle timing reporting reference",
            ["SRC-SIVAK-2023"],
            "main_digital_twin",
            "NUMERIC_REFERENCE",
            {"gain": "2.27+/-0.07", "constituent_us": 4.924, "full_cycle_us": 9.848},
            {"kind": "reference_only", "project_pass_fail_uses_same_model_and_cost_only": True},
            "reference_only_no_model_selection",
            False,
            "REFERENCE_ONLY",
            {"project_device_timing_measured": False},
            [],
            "T5.1.5 physical-time fairness and T6 timing measurement",
            ["literature gain is not a simulator target", "literature timing is not target-board timing"],
        ),
        _target(
            "LT-2025-NMF-DIRECTION",
            "Puviani PRL NMF versus registered MF and standard direction",
            ["SRC-PUVIANI-PRL-2025"],
            "main_digital_twin",
            "DIRECTIONAL",
            {"ordering": "NMF_above_MF_above_or_equal_standard_in_registered_PRL_like_lane", "paper_scale_lifetime_gain": "about_100_percent_reference_only"},
            {"kind": "registered_model_specific_holdout", "paired_nmf_minus_mf_logical_z_lifetime_CI_low_min": 0.0},
            "independent_holdout",
            False,
            "QUALIFIED_DIRECTIONAL_PASS",
            {"mean_difference": paired["mean_difference"], "ci95_low": paired["ci95_low"], "ci95_high": paired["ci95_high"], "later_exact_budget_MF_ordering_reverses_across_cutoffs": counter["ordering_reverses_across_cutoffs"]},
            [str(ARTIFACT_BINDINGS["T2.3.7"]), str(ARTIFACT_BINDINGS["T4.4.5"])],
            "T5.4.4 exact-budget multi-agent audit and T5.4.5 long horizon",
            ["no universal NMF-over-MF claim", "no paper-amplitude reproduction claim", "no optimizer or device claim"],
        ),
        _target(
            "LT-2025-NMF-HORIZON",
            "Puviani PRL 10-cycle training to 1000-cycle evaluation extrapolation",
            ["SRC-PUVIANI-PRL-2025"],
            "main_digital_twin",
            "LONG_HORIZON_DIRECTIONAL",
            {"training_cycles": 10, "paper_evaluation_cycles": 1000, "required_project_sweeps": [1000, 100000, 1000000]},
            {"kind": "future_holdout", "criterion": "bounded_hidden_state_and_preregistered_gain_retention_with_numerical_stability"},
            "future_holdout_preregistered",
            False,
            "REGISTERED_PENDING",
            {"project_long_horizon_executed": False},
            [],
            "T5.4.5 horizon extrapolation",
            ["ten-cycle success cannot be extrapolated in prose", "paper 1500-cycle lifetime is not a project target"],
        ),
        _target(
            "LT-2025-KNILL-EQUIVALENCE",
            "Knill/Steane special-case numerical equivalence",
            ["SRC-MARQVERSEN-2025"],
            "secondary_reproduction",
            "NUMERIC_REGRESSION",
            {"max_absolute_equivalence_error": 1e-8},
            {"kind": "future_secondary_holdout", "max_absolute_error": 1e-8, "independent_parameter_grid": True},
            "future_holdout_preregistered",
            False,
            "REGISTERED_PENDING",
            {"secondary_protocol_executable": False},
            [str(ARTIFACT_BINDINGS["T2.2.2"])],
            "T5.0.2 secondary analytic holdout",
            ["never enter sBs main ranking", "no FPGA physical-control claim"],
        ),
        _target(
            "LT-2025-QUNAUGHT-SQUEEZING",
            "Qunaught-resource Knill squeezing trend",
            ["SRC-MARQVERSEN-2025"],
            "secondary_reproduction",
            "DIRECTIONAL",
            {"direction": "qunaught_Knill_preserves_superior_symmetric_GKP_squeezing_among_registered_variants"},
            {"kind": "future_secondary_holdout", "all_preregistered_squeezing_points_must_preserve_direction": True},
            "future_holdout_preregistered",
            False,
            "REGISTERED_PENDING",
            {"secondary_protocol_executable": False},
            [str(ARTIFACT_BINDINGS["T2.2.2"])],
            "T5.0.2 secondary analytic holdout",
            ["never enter sBs main ranking", "no optical-simplicity transfer to current hardware"],
        ),
        _target(
            "LT-2026-PSTEANE-CONDITION",
            "P-Steane small-noise optimum condition",
            ["SRC-CHEN-2026"],
            "secondary_reproduction",
            "ANALYTIC_AND_NUMERIC",
            {"condition": "2a=b", "metric": "product_of_q_and_p_output_noise_variances"},
            {"kind": "future_secondary_holdout", "analytic_stationarity_required": True, "numeric_argmin_grid_error_max": 0.01},
            "future_holdout_preregistered",
            False,
            "REGISTERED_PENDING",
            {"secondary_protocol_executable": False},
            [str(ARTIFACT_BINDINGS["T2.2.2"])],
            "T5.0.2 secondary analytic holdout",
            ["condition only applies in stated small-noise/data-noisier regime", "never enter sBs main ranking", "FPGA may select parameters but does not implement physical squeezing"],
        ),
        _target(
            "LT-2026-PSTEANE-NOISE-RATIO",
            "P-Steane data-to-ancilla noise-ratio shaping trend",
            ["SRC-CHEN-2026"],
            "secondary_reproduction",
            "DIRECTIONAL",
            {"direction": "benefit_and_selected_a_b_change_with_data_to_ancilla_noise_ratio", "special_cases": {"ME_Steane": [1.0, 1.0], "teleportation": [0.7071067811865476, 1.4142135623730951]}},
            {"kind": "future_secondary_holdout", "disjoint_noise_ratio_grid": True, "compare_against_ME_Steane": True},
            "future_holdout_preregistered",
            False,
            "REGISTERED_PENDING",
            {"secondary_protocol_executable": False},
            [str(ARTIFACT_BINDINGS["T2.2.2"])],
            "T5.0.2 secondary analytic holdout",
            ["never enter sBs main ranking", "no universal P-Steane superiority outside source regime"],
        ),
        _target(
            "LT-2024-NOISE-TRANSFER-HIGH",
            "Noise-transfer localized high-squeezing validity domain",
            ["SRC-RALPH-2024"],
            "main_digital_twin",
            "VALIDITY_BOUNDARY",
            {"squeezing_db_min": 10.0, "localized_peak_assumption": True},
            {"kind": "registered_cross_fidelity", "noise_vs_direct_q_ler_gap_max": 5e-5, "effective_noise_z_score_max": 2.0, "canonical_qp_gap_max": 1e-6},
            "calibration_only",
            True,
            "QUALIFIED_DIRECTIONAL_PASS",
            {"noise_vs_direct_q_ler_gap": t233["maximum_high_squeezing_noise_syndrome_q_ler_gap"], "effective_noise_z_score": t233["maximum_high_squeezing_effective_noise_z_score"], "canonical_qp_gap": t233["maximum_high_squeezing_canonical_fock_qp_ler_gap"]},
            [str(ARTIFACT_BINDINGS["T2.3.3"])],
            "T5.0.2 independent cross-fidelity holdout",
            ["validity is local to high squeezing", "no coherent joint-axis or device-fidelity claim"],
        ),
        _target(
            "LT-2024-NOISE-TRANSFER-LOW",
            "Noise-transfer low-squeezing clipping failure boundary",
            ["SRC-RALPH-2024"],
            "main_digital_twin",
            "NEGATIVE_VALIDITY_BOUNDARY",
            {"squeezing_db": 3.0, "expected": "clipping_dominated_mismatch"},
            {"kind": "registered_negative_control", "noise_vs_direct_q_ler_gap_min": 0.01, "clipping_ratio_max": 0.5},
            "calibration_only",
            True,
            "NEGATIVE_BOUNDARY_VERIFIED",
            {"noise_vs_direct_q_ler_gap": t233["low_squeezing_noise_syndrome_q_ler_gap"], "minimum_clipping_ratio": low_point["noise_transfer"]["minimum_clipping_ratio"]},
            [str(ARTIFACT_BINDINGS["T2.3.3"])],
            "T5.0.2 retain low-squeezing negative holdout",
            ["do not repair the mismatch by retuning on holdout", "do not use the surrogate below its validity domain"],
        ),
        _target(
            "LT-2026-TRAPPED-ION-REPORT",
            "Trapped-ion two-mode GKP QEC-on/off reporting structure",
            ["SRC-FONTBOTE-2026"],
            "secondary_reproduction",
            "REPORTING_TEMPLATE",
            {"required_fields": ["Pauli_resolved_on_off_lifetimes", "ratio_with_uncertainty", "wall_clock_per_round", "reset_recoil", "parallel_control_cost"], "external_context": {"Bell_fidelity": "0.69(1)", "Pauli_lifetime_ms": {"XX": {"on": "5.0(7)", "off": "2.4(2)", "ratio": "2.1(3)"}, "YY": {"on": "3.8(9)", "off": "2.3(4)", "ratio": "1.7(5)"}, "ZZ": {"on": "5.3(8)", "off": "2.3(2)", "ratio": "2.3(4)"}}, "mean_lifetime_gain": "2.0(2)", "round_us": 500}},
            {"kind": "report_schema_only", "numeric_values_are_not_acceptance_thresholds": True},
            "reporting_template_only",
            False,
            "REPORTING_TEMPLATE_ONLY",
            {"project_two_mode_trapped_ion_experiment": False},
            [],
            "T5.3.1/T5.3.4 project-native channel and cost reporting",
            ["two-mode trapped-ion numbers cannot be transferred to single-mode sBs", "not a hardware-reproduction row", "not part of sBs ranking"],
        ),
    ]


def _targets_complete(targets: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        "target_id", "topic", "source_ids", "hierarchy_role", "target_type",
        "reference_target", "tolerance_rule", "calibration_or_holdout_use",
        "model_selection_access", "current_status", "current_observation",
        "evidence_artifacts", "next_gate", "prohibited_transfer",
    }
    return bool(targets) and all(required == set(row) and all(row[key] not in (None, "", []) for key in ("target_id", "topic", "source_ids", "hierarchy_role", "target_type", "reference_target", "tolerance_rule", "calibration_or_holdout_use", "current_status", "next_gate", "prohibited_transfer")) for row in targets)


def build_registry(
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    loaded = dict(artifacts or load_bound_artifacts())
    missing = set(ARTIFACT_BINDINGS) - set(loaded)
    if missing:
        raise ValueError(f"missing bound artifacts: {sorted(missing)}")
    targets = build_targets(loaded)
    anchors = inspect_local_anchors()
    bindings = {
        task_id: {
            "path": str(path),
            "sha256": _sha256(path),
            "machine_pass": _artifact_machine_pass(task_id, loaded[task_id]),
        }
        for task_id, path in ARTIFACT_BINDINGS.items()
    }
    topics = {row["target_id"] for row in targets}
    secondary = [row for row in targets if row["hierarchy_role"] == "secondary_reproduction"]
    reference_like = [row for row in targets if row["current_status"] in {"REFERENCE_ONLY", "REPORTING_TEMPLATE_ONLY"}]
    holdouts = [row for row in targets if row["calibration_or_holdout_use"] in {"independent_holdout", "future_holdout_preregistered"}]
    t233 = loaded["T2.3.3"]
    t445 = loaded["T4.4.5"]
    gates = {
        "fourteen_unique_targets_registered": len(targets) == len(topics) == 14,
        "all_required_topics_covered": all(any(token in row["topic"] for row in targets) for token in ("Campagne", "Sivak", "Puviani", "Knill", "Qunaught", "P-Steane", "Noise-transfer", "Trapped-ion")),
        "every_target_has_complete_contract": _targets_complete(targets),
        "all_current_status_values_are_controlled": all(row["current_status"] in STATUS_VALUES for row in targets),
        "all_official_sources_have_identifier_url_and_verification_date": all(source["identifier"] and source["official_url"].startswith("https://") and source["verified_on"] == "2026-07-16" for source in SOURCES),
        "all_local_source_anchors_match": bool(anchors) and all(row["passed"] for row in anchors),
        "all_bound_artifacts_are_current_and_machine_pass": all(row["machine_pass"] for row in bindings.values()),
        "calibration_holdout_and_reference_use_are_explicit": {row["calibration_or_holdout_use"] for row in targets} >= {"calibration_only", "independent_holdout", "future_holdout_preregistered", "reference_only_no_model_selection", "reporting_template_only"},
        "holdout_and_reference_rows_cannot_select_models": all(row["model_selection_access"] is False for row in holdouts + reference_like),
        "secondary_rows_are_excluded_from_sbs_main_ranking": bool(secondary) and all(any("sBs" in text or "main ranking" in text for text in row["prohibited_transfer"]) for row in secondary),
        "pending_rows_are_not_marked_as_reproduced": all(row["current_status"] == "REGISTERED_PENDING" for row in targets if row["current_observation"].get("secondary_protocol_executable") is False),
        "high_squeezing_noise_transfer_gate_matches_current_artifact": bool(t233["maximum_high_squeezing_noise_syndrome_q_ler_gap"] <= 5e-5 and t233["maximum_high_squeezing_effective_noise_z_score"] <= 2.0 and t233["maximum_high_squeezing_canonical_fock_qp_ler_gap"] <= 1e-6),
        "low_squeezing_clipping_negative_boundary_is_preserved": bool(t233["low_squeezing_noise_syndrome_q_ler_gap"] >= 0.01 and next(point for point in t233["points"] if float(point["squeezing_db"]) == 3.0)["noise_transfer"]["minimum_clipping_ratio"] < 0.5),
        "nmf_exact_budget_cutoff_reversal_is_preserved": t445["counterevidence"]["ordering_reverses_across_cutoffs"] is True,
        "external_numeric_references_are_not_project_acceptance_targets": all(row["model_selection_access"] is False for row in targets if row["current_status"] in {"REFERENCE_ONLY", "REPORTING_TEMPLATE_ONLY"}),
        "table_completion_does_not_claim_all_reproductions_pass": any(row["current_status"] == "REGISTERED_PENDING" for row in targets),
        "reporting_template_is_nontransferable_secondary_evidence": next(row for row in targets if row["target_id"] == "LT-2026-TRAPPED-ION-REPORT")["hierarchy_role"] == "secondary_reproduction",
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": implementation_sha256(),
        "completion_semantics": "PASS means the reproduction registry is complete and internally current; it does not mean every registered literature trend has been reproduced.",
        "sources": list(SOURCES),
        "local_source_anchors": anchors,
        "artifact_bindings": bindings,
        "targets": targets,
        "coverage_summary": {
            "target_count": len(targets),
            "by_status": {status_value: sum(row["current_status"] == status_value for row in targets) for status_value in sorted(STATUS_VALUES)},
            "by_hierarchy_role": {role: sum(row["hierarchy_role"] == role for row in targets) for role in sorted({row["hierarchy_role"] for row in targets})},
            "current_pass_like_count": sum(row["current_status"] in {"QUALIFIED_DIRECTIONAL_PASS", "NEGATIVE_BOUNDARY_VERIFIED", "STRUCTURE_IMPLEMENTED_NOT_NUMERIC_REPRODUCTION"} for row in targets),
            "pending_count": sum(row["current_status"] == "REGISTERED_PENDING" for row in targets),
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "failed": [name for name, value in gates.items() if not value],
        },
        "claim_boundary": {
            "allowed": "source-bound trend registry plus explicitly qualified current model evidence",
            "forbidden": "claiming all rows reproduced, transferring secondary protocols into sBs ranking, matching cross-platform lifetimes/timing, or using holdout/reference rows for model selection",
        },
    }
    payload["registry_contract_sha256"] = _canonical_sha256({key: value for key, value in payload.items() if key not in {"generated_at_utc", "registry_contract_sha256"}})
    return payload


def source_data_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in payload["sources"]:
        rows.append({"row_type": "source", "record_id": source["source_id"], "topic": source["title"], "hierarchy_role": source["source_class"], "target_type": "", "status": "VERIFIED_METADATA", "numeric_value": source["year"], "tolerance": source["identifier"], "use": "reference_source", "artifact": source["official_url"], "passed": True, "detail": source["verified_on"]})
    for anchor in payload["local_source_anchors"]:
        rows.append({"row_type": "source_anchor", "record_id": anchor["anchor_id"], "topic": anchor["source_id"], "hierarchy_role": "local_primary_text", "target_type": "line_fragment", "status": "CURRENT" if anchor["passed"] else "STALE", "numeric_value": anchor["line"], "tolerance": "exact_fragment", "use": "source_provenance", "artifact": anchor["path"], "passed": anchor["passed"], "detail": anchor["fragment"]})
    for task_id, binding in payload["artifact_bindings"].items():
        rows.append({"row_type": "artifact_binding", "record_id": task_id, "topic": "current project evidence", "hierarchy_role": "project_artifact", "target_type": "sha256", "status": "PASS" if binding["machine_pass"] else "FAIL", "numeric_value": "", "tolerance": "all machine gates true", "use": "current_evidence", "artifact": binding["path"], "passed": binding["machine_pass"], "detail": binding["sha256"]})
    for target in payload["targets"]:
        rows.append({"row_type": "target", "record_id": target["target_id"], "topic": target["topic"], "hierarchy_role": target["hierarchy_role"], "target_type": target["target_type"], "status": target["current_status"], "numeric_value": json.dumps(target["current_observation"], sort_keys=True, ensure_ascii=False), "tolerance": json.dumps(target["tolerance_rule"], sort_keys=True, ensure_ascii=False), "use": target["calibration_or_holdout_use"], "artifact": ";".join(target["evidence_artifacts"]), "passed": target["current_status"] in {"QUALIFIED_DIRECTIONAL_PASS", "NEGATIVE_BOUNDARY_VERIFIED", "STRUCTURE_IMPLEMENTED_NOT_NUMERIC_REPRODUCTION"}, "detail": "; ".join(target["prohibited_transfer"])})
    for gate_id, passed in payload["gates"].items():
        rows.append({"row_type": "gate", "record_id": gate_id, "topic": "registry completion", "hierarchy_role": "machine_gate", "target_type": "boolean", "status": "PASS" if passed else "FAIL", "numeric_value": "", "tolerance": "must_be_true", "use": "task_acceptance", "artifact": str(DEFAULT_ARTIFACT), "passed": passed, "detail": ""})
    return rows


def write_artifacts(
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = build_registry()
    rows = source_data_rows(payload)
    csv_path = _repo_path(source_data_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("row_type", "record_id", "topic", "hierarchy_role", "target_type", "status", "numeric_value", "tolerance", "use", "artifact", "passed", "detail")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    payload["source_data"] = {"path": str(Path(source_data_path)), "row_count": len(rows), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    output = _repo_path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE_DATA))
    args = parser.parse_args(argv)
    payload = write_artifacts(args.artifact, args.source_data)
    print(json.dumps({"task_id": TASK_ID, "status": payload["status"], "gates": payload["gate_summary"], "targets": len(payload["targets"]), "source_rows": payload["source_data"]["row_count"]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
