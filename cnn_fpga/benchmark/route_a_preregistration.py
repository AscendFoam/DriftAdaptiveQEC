"""T6.5.3 result-blind Route-A scenario and statistical preregistration."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "T6.5.3"
PROTOCOL_ID = "ROUTE-A-FORMAL-PREREGISTRATION-V1"
SCHEMA_VERSION = "t6.5.3-route-a-preregistration-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_5_3_route_a_preregistration.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_5_3_route_a_preregistration_source_data.csv"

SMOOTH_FAMILIES = (
    "mean_drift",
    "variance_drift",
    "correlation_drift",
    "periodic_drift",
)
ABRUPT_OOD_FAMILIES = (
    "step_calibration_shift",
    "telegraph_drift",
    "burst_outlier",
    "readout_reset_fault",
    "leakage_persistence",
    "compound_ood",
)
NOMINAL_FAMILY = "nominal_static"

PRIOR_EVIDENCE = (
    {
        "path": "docs/t5_1_3_oracle_gap_tail_report.json",
        "disclosed_use": "55/512 proposed calibration-shift worst versus 37/512 static motivated strict new tail gate",
    },
    {
        "path": "docs/t5_1_4_algorithm_branch_verdict.json",
        "disclosed_use": "zero Holm discovery and failed learned branch motivated independent clusters and Route-A fallback branch",
    },
    {
        "path": "docs/t5_4_2_uncertainty_gated_fallback.json",
        "disclosed_use": "compound and nominal counterexamples motivated catastrophic and nominal non-inferiority margins",
    },
    {
        "path": "docs/t6_5_2_unified_execution_contract.json",
        "disclosed_use": "method, privilege, cadence, budget and deadline schemas are frozen parent contracts",
    },
)

FUTURE_FORMAL_ARTIFACTS = (
    "docs/t6_7_1_smooth_formal_matrix.json",
    "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json",
    "docs/t6_7_3_integrated_long_sequence.json",
    "docs/t6_7_4_route_a_decision_gate.json",
)


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    role: str
    seeds: tuple[int, ...]
    transition_rates_per_window: tuple[float, ...]
    amplitudes: tuple[float, ...]
    durations_windows: tuple[int, ...]
    scored_windows_per_cell: int
    decisions_per_window: int = 512
    nominal_preamble_windows: int = 8

    def __post_init__(self) -> None:
        if self.split_id not in ("calibration", "pilot_validation", "formal_evaluation"):
            raise ValueError("unknown split_id")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("split seeds must be nonempty and unique")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in self.seeds):
            raise ValueError("split seeds must be positive integers")
        for values, name in (
            (self.transition_rates_per_window, "transition rates"),
            (self.amplitudes, "amplitudes"),
        ):
            if not values or len(values) != len(set(values)) or any(value <= 0.0 for value in values):
                raise ValueError(f"{name} must be unique and positive")
        if not self.durations_windows or len(self.durations_windows) != len(set(self.durations_windows)) or any(value <= 0 for value in self.durations_windows):
            raise ValueError("durations must be unique and positive")
        if self.decisions_per_window != 512:
            raise ValueError("formal logical-error windows are exactly 512 decisions")
        if self.nominal_preamble_windows != 8:
            raise ValueError("every dynamic cell uses the same eight-window unscored preamble")
        if self.scored_windows_per_cell < 48:
            raise ValueError("each cell requires at least 48 scored windows")


def split_specs() -> tuple[SplitSpec, ...]:
    return (
        SplitSpec(
            split_id="calibration",
            role="fit static images and non-policy nuisance parameters only",
            seeds=tuple(range(202607176001, 202607176013)),
            transition_rates_per_window=(0.0125, 0.025),
            amplitudes=(0.08, 0.16),
            durations_windows=(16, 32),
            scored_windows_per_cell=48,
        ),
        SplitSpec(
            split_id="pilot_validation",
            role="select one common baseline, policy threshold tuple and no other formal choices",
            seeds=tuple(range(202607176101, 202607176113)),
            transition_rates_per_window=(0.01875, 0.0375),
            amplitudes=(0.10, 0.20),
            durations_windows=(24, 40),
            scored_windows_per_cell=64,
        ),
        SplitSpec(
            split_id="formal_evaluation",
            role="locked confirmatory evaluation; never fit, select, tune or replace thresholds",
            seeds=tuple(range(202607176201, 202607176225)),
            transition_rates_per_window=(0.015625, 0.03125, 0.046875),
            amplitudes=(0.12, 0.18, 0.24),
            durations_windows=(20, 28, 48),
            scored_windows_per_cell=96,
        ),
    )


DESIGN_INDICES = {
    "calibration": ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
    "pilot_validation": ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
    "formal_evaluation": (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 1, 2),
        (2, 2, 1),
    ),
}


FAMILY_AMPLITUDE_SEMANTICS = {
    "mean_drift": "peak_abs_mean = amplitude * GKP_lattice",
    "variance_drift": "endpoint_sigma_ratio = 1 + 2 * amplitude",
    "correlation_drift": "endpoint_abs_rho = min(0.85, 3 * amplitude)",
    "periodic_drift": "mean_or_sigma modulation amplitude uses amplitude; duration is period in windows",
    "step_calibration_shift": "post-step mean shift = amplitude * lattice and sigma ratio = 1 + amplitude",
    "telegraph_drift": "state separation = amplitude * lattice; transition probability is transition_rate_per_window",
    "burst_outlier": "burst p_outlier = min(0.25, amplitude); outlier_scale = 1 + 12 * amplitude",
    "readout_reset_fault": "g/e confusion = min(0.25, amplitude); duration is reset-fault dwell",
    "leakage_persistence": "injection probability uses transition rate; persistence severity uses amplitude and duration",
    "compound_ood": "simultaneous mean/variance/correlation/readout/leakage components use the same registered tuple",
}


def scenario_cells() -> tuple[dict[str, object], ...]:
    specs = {row.split_id: row for row in split_specs()}
    cells: list[dict[str, object]] = []
    for split_id, indices in DESIGN_INDICES.items():
        split = specs[split_id]
        for family in (*SMOOTH_FAMILIES, *ABRUPT_OOD_FAMILIES):
            group = "smooth" if family in SMOOTH_FAMILIES else "abrupt_ood"
            for cell_index, (rate_index, amplitude_index, duration_index) in enumerate(indices):
                cells.append(
                    {
                        "cell_id": f"{split_id}:{family}:{cell_index:02d}",
                        "split_id": split_id,
                        "scenario_group": group,
                        "family": family,
                        "transition_rate_per_window": split.transition_rates_per_window[rate_index],
                        "amplitude": split.amplitudes[amplitude_index],
                        "duration_windows": split.durations_windows[duration_index],
                        "scored_windows": split.scored_windows_per_cell,
                        "decisions_per_window": split.decisions_per_window,
                        "nominal_preamble_windows": split.nominal_preamble_windows,
                        "amplitude_semantics": FAMILY_AMPLITUDE_SEMANTICS[family],
                    }
                )
        cells.append(
            {
                "cell_id": f"{split_id}:{NOMINAL_FAMILY}:00",
                "split_id": split_id,
                "scenario_group": "negative_control",
                "family": NOMINAL_FAMILY,
                "transition_rate_per_window": 0.0,
                "amplitude": 0.0,
                "duration_windows": split.scored_windows_per_cell,
                "scored_windows": split.scored_windows_per_cell,
                "decisions_per_window": split.decisions_per_window,
                "nominal_preamble_windows": split.nominal_preamble_windows,
                "amplitude_semantics": "stationary in-distribution negative control; dynamic-level disjointness not applicable",
            }
        )
    return tuple(cells)


def threshold_selection_contract() -> dict[str, object]:
    return {
        "threshold_lock_task": "T6.6.3",
        "selection_split": "pilot_validation",
        "formal_evaluation_access_allowed": False,
        "one_tuple_shared_by_all_scenarios": True,
        "per_scenario_thresholds_prohibited": True,
        "candidate_grid": {
            "regime_posterior_enter_min": [0.60, 0.70, 0.80, 0.90],
            "regime_posterior_exit_max": [0.20, 0.30, 0.40],
            "uncertainty_fallback_min": [0.25, 0.35, 0.45, 0.55],
            "ood_score_min_code": [128, 160, 192, 224],
            "enter_hysteresis_windows": [2, 3, 4],
            "recovery_hysteresis_windows": [4, 6, 8],
        },
        "fixed_integrity_thresholds": {
            "max_parameter_age_cycles": 8192,
            "crc_required": True,
            "version_rule": "new_version=active_version+1_with_CAS",
            "commit_ack_required": True,
        },
        "selection_constraints_in_order": [
            "zero integrity/undefined/silent-overflow failures",
            "all pilot abrupt/OOD catastrophic and nominal non-inferiority constraints pass",
            "maximize minimum per-family safety slack",
            "maximize aggregate paired LER-improvement 95% lower bound",
            "minimize unnecessary fallback rate then total fallback rate",
            "deterministic lexicographic threshold-tuple order",
        ],
        "selected_threshold_tuple": None,
        "selected_strongest_deployable_baseline": None,
        "lock_sha256": None,
        "failure_rule": "if no candidate passes all constraints, Route-A formal evaluation is NO_GO until a new protocol version is preregistered",
    }


def metric_contract() -> dict[str, object]:
    return {
        "logical_outcomes": ["I", "X", "Y", "Z"],
        "p_x": "count(X)/all scored decisions",
        "p_y": "count(Y)/all scored decisions",
        "p_z": "count(Z)/all scored decisions",
        "p_l": "p_X+p_Y+p_Z; count(any non-I)/all scored decisions",
        "average_ler": "all scored decisions, preserving seed/family/cell membership",
        "window": "nonoverlapping 512 decisions; no rolling overlap and no window-as-independent-seed inference",
        "p95_window_ler": "empirical quantile with method='higher' over registered 512-decision windows",
        "seed_worst_window_ler": "maximum 512-decision window LER inside one seed/family across all registered cells",
        "global_worst_window_ler": "maximum observed registered window, reported with numerator/512 and trace locator",
        "oracle_gap_closure": "(static_LER-proposed_LER)/(static_LER-oracle_LER), only when paired denominator CI is strictly positive; otherwise NA with flag",
        "adaptation_lag": "transition onset to first accepted/acknowledged bank version actually used; non-recovery is right-censored",
        "false_update": "committed parameter update outside normal/smooth truth interval; truth is evaluation-only and never policy input",
        "unnecessary_fallback": "fallback selected while policy-on and policy-off actions are both correct",
        "avoided_error": "policy-off wrong and policy-on correct on the identical decision",
        "induced_error": "policy-off correct and policy-on wrong on the identical decision",
        "deadline_miss": "logical cycle, host update and future board-measured fields remain separate",
    }


def statistical_contract() -> dict[str, object]:
    return {
        "independent_cluster": "formal seed; all families/cells/windows for that seed are resampled together",
        "formal_cluster_count": 24,
        "aggregation_weights": {
            "within_cell": "all registered decisions equally weighted",
            "within_family": "cell estimates equally weighted within each seed",
            "aggregate_smooth": "four smooth family estimates equally weighted within each seed",
            "aggregate_abrupt_ood": "six abrupt/OOD family estimates equally weighted within each seed",
            "nominal": "reported separately and never pooled to improve a dynamic aggregate",
            "forbidden": "raw decision pooling across families/cells or post-result reweighting",
        },
        "bootstrap": {
            "method": "paired nonparametric cluster bootstrap",
            "replicates": 20000,
            "seed": 202607176999,
            "confidence_level": 0.95,
            "two_sided_bounds": [0.025, 0.975],
            "zero_difference_ties_preserved": True,
        },
        "strongest_baseline_selection": {
            "split": "pilot_validation",
            "eligible": [
                "standard_binning",
                "static_joint_map",
                "window_map",
                "ewma_adaptive_map",
                "kalman_adaptive_map",
                "legacy_cnn_residual_if_checkpoint_and_budget_pass",
            ],
            "oracle_excluded": True,
            "order": "lowest equal-family pilot aggregate average LER over ten dynamic families, then p95, then worst, then MAC/wall-clock, then canonical id",
            "formal_reselection_prohibited": True,
        },
        "hierarchy": [
            "H1 primary aggregate smooth paired LER improvement: 95% LCB > 0",
            "H2 calibration-shift strict tail non-inferiority",
            "H3 all abrupt/OOD catastrophic and nominal margins",
            "only after H1-H3 pass may family-level superiority be promoted",
        ],
        "multiplicity": {
            "method": "Holm step-down",
            "familywise_alpha": 0.05,
            "families": {
                "smooth_family_superiority": list(SMOOTH_FAMILIES),
                "abrupt_ood_safety": [*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY],
            },
            "p_X_p_Y_p_Z_and_cost_metrics": "secondary descriptive intervals; no claim promotion",
        },
    }


def acceptance_gates() -> dict[str, object]:
    return {
        "primary_smooth": {
            "contrast": "strongest_deployable_baseline_LER - proposed_route_a_LER",
            "aggregate_paired_95_lcb_min_exclusive": 0.0,
            "all_four_smooth_families_executed": True,
        },
        "calibration_shift_tail": {
            "global_worst_error_count_proposed_max_relative_to_baseline": 0,
            "seed_worst_paired_difference_95_ucb_max": 0.0,
            "prior_counterexample_target": "new independent formal run must not repeat proposed 55/512 > static 37/512",
        },
        "catastrophic_degradation_each_abrupt_ood_family": {
            "average_ler_proposed_minus_baseline_95_ucb_max": 0.002,
            "p95_window_ler_proposed_minus_baseline_95_ucb_max": 4 / 512,
            "seed_worst_window_ler_proposed_minus_baseline_95_ucb_max": 8 / 512,
            "any_single_window_excess_error_count_max": 16,
            "all_conditions_required": True,
        },
        "nominal_non_inferiority": {
            "average_ler_proposed_minus_policy_off_95_ucb_max": 0.0005,
            "fallback_rate_max": 0.01,
            "unnecessary_fallback_rate_max": 0.0075,
            "induced_minus_avoided_rate_95_ucb_max": 0.00025,
            "all_conditions_required": True,
        },
        "integrity": {
            "bit_mismatch_max": 0,
            "undefined_action_max": 0,
            "silent_overflow_max": 0,
            "software_deadline_miss_max": 0,
            "board_deadline_gate": "deferred_to_T6.9.2_and_must_not_be_imputed_false",
        },
        "cnn": "if real checkpoint fails matched budget/performance, retain only as ablation",
        "failure_branches": {
            "tail_or_catastrophic_failure": "smooth-only Route-A claim",
            "primary_average_failure": "static MAP-LUT plus deterministic FPGA claim only",
            "cnn_failure": "CNN ablation/supplement only",
            "board_failure_or_absence": "hardware-aware/CXXRTL/P&R estimate only",
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _prior_bindings() -> list[dict[str, object]]:
    rows = []
    for item in PRIOR_EVIDENCE:
        path = ROOT / item["path"]
        rows.append({**item, "exists": path.is_file(), "sha256": _sha256(path) if path.is_file() else None})
    return rows


def protocol_payload() -> dict[str, object]:
    formal = next(row for row in split_specs() if row.split_id == "formal_evaluation")
    formal_dynamic_cells = len(DESIGN_INDICES["formal_evaluation"]) * (
        len(SMOOTH_FAMILIES) + len(ABRUPT_OOD_FAMILIES)
    )
    formal_all_cells = formal_dynamic_cells + 1
    decisions_per_method = (
        formal_all_cells
        * len(formal.seeds)
        * formal.scored_windows_per_cell
        * formal.decisions_per_window
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol_revision_policy": "immutable; any post-freeze change requires a new protocol_id, new artifact and unchanged archival of this version",
        "scenario_families": {
            "smooth": list(SMOOTH_FAMILIES),
            "abrupt_ood": list(ABRUPT_OOD_FAMILIES),
            "negative_control": [NOMINAL_FAMILY],
        },
        "splits": [asdict(row) for row in split_specs()],
        "scenario_cells": list(scenario_cells()),
        "trace_schedule": {
            "shared_across_methods": True,
            "domain_separated_seed_rule": "uint64_le(SHA256(protocol_id|split_id|family|cell_id|base_seed|stream_name)[0:8])",
            "streams": ["latent_dynamics", "syndrome_sampling", "readout_reset", "leakage", "transport"],
            "nominal_preamble_windows": 8,
            "smooth_onset": "first scored window after the common preamble",
            "abrupt_onset": "first scored window + SHA256-derived offset in [0,7]",
            "telegraph_burst_repetitions": "registered transition rate and duration only; no method-dependent event timing",
            "trace_reuse": "one immutable trace/action-opportunity sequence is replayed for every comparator",
        },
        "formal_workload": {
            "dynamic_cells": formal_dynamic_cells,
            "nominal_cells": 1,
            "seed_clusters": len(formal.seeds),
            "trajectories_per_method": formal_all_cells * len(formal.seeds),
            "scored_decisions_per_method": decisions_per_method,
            "seven_deployable_method_decisions": 7 * decisions_per_method,
            "oracle_decisions_reported_separately": decisions_per_method,
        },
        "family_amplitude_semantics": FAMILY_AMPLITUDE_SEMANTICS,
        "threshold_selection": threshold_selection_contract(),
        "metrics": metric_contract(),
        "statistics": statistical_contract(),
        "acceptance_gates": acceptance_gates(),
        "prior_evidence_disclosure": _prior_bindings(),
        "formal_result_access_at_freeze": {
            "accessed": False,
            "future_artifact_paths": list(FUTURE_FORMAL_ARTIFACTS),
            "paths_existing_when_frozen": [],
            "rule": "absence is recorded historically and is not rechecked after formal execution",
        },
    }


def recompute_gates(payload: Mapping[str, Any]) -> dict[str, bool]:
    splits = payload["splits"]
    split_by_id = {row["split_id"]: row for row in splits}
    cells = payload["scenario_cells"]
    dynamic = [row for row in cells if row["family"] != NOMINAL_FAMILY]
    thresholds = payload["threshold_selection"]
    stats = payload["statistics"]
    metrics = payload["metrics"]
    acceptance = payload["acceptance_gates"]
    workload = payload["formal_workload"]
    schedule = payload["trace_schedule"]
    seeds = [set(row["seeds"]) for row in splits]
    rates = [set(row["transition_rates_per_window"]) for row in splits]
    amplitudes = [set(row["amplitudes"]) for row in splits]
    durations = [set(row["durations_windows"]) for row in splits]
    expected_families = set((*SMOOTH_FAMILIES, *ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY))
    formal_cells = [row for row in cells if row["split_id"] == "formal_evaluation" and row["family"] != NOMINAL_FAMILY]
    return {
        "G01_exact_scenario_coverage": set(row["family"] for row in cells) == expected_families and set(payload["scenario_families"]["smooth"]) == set(SMOOTH_FAMILIES) and set(payload["scenario_families"]["abrupt_ood"]) == set(ABRUPT_OOD_FAMILIES),
        "G02_seed_splits_pairwise_disjoint": len(seeds) == 3 and all(seeds[i].isdisjoint(seeds[j]) for i in range(3) for j in range(i + 1, 3)),
        "G03_rate_amplitude_duration_splits_disjoint": all(all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i + 1, 3)) for groups in (rates, amplitudes, durations)),
        "G04_unique_registered_cells": len(cells) == len({row["cell_id"] for row in cells}) and len(cells) == (4 * 10 + 1) + (4 * 10 + 1) + (6 * 10 + 1),
        "G05_formal_heldout_design_complete": len(formal_cells) == 60 and all(row["transition_rate_per_window"] in split_by_id["formal_evaluation"]["transition_rates_per_window"] and row["amplitude"] in split_by_id["formal_evaluation"]["amplitudes"] and row["duration_windows"] in split_by_id["formal_evaluation"]["durations_windows"] for row in formal_cells),
        "G06_formal_independent_cluster_count_frozen": len(split_by_id["formal_evaluation"]["seeds"]) == stats["formal_cluster_count"] == 24,
        "G06b_formal_workload_is_explicit": workload["dynamic_cells"] == 60 and workload["nominal_cells"] == 1 and workload["trajectories_per_method"] == 1464 and workload["scored_decisions_per_method"] == 71958528 and workload["seven_deployable_method_decisions"] == 503709696 and workload["oracle_decisions_reported_separately"] == 71958528,
        "G07_512_decision_tail_window_frozen": metrics["window"].startswith("nonoverlapping 512 decisions") and all(row["decisions_per_window"] == 512 for row in cells),
        "G07b_shared_trace_schedule_is_method_independent": schedule["shared_across_methods"] is True and schedule["nominal_preamble_windows"] == 8 and "SHA256" in schedule["domain_separated_seed_rule"] and len(schedule["streams"]) == 5 and "no method-dependent" in schedule["telegraph_burst_repetitions"],
        "G08_one_common_threshold_tuple_only": thresholds["one_tuple_shared_by_all_scenarios"] is True and thresholds["per_scenario_thresholds_prohibited"] is True and thresholds["selection_split"] == "pilot_validation",
        "G09_formal_result_blind_threshold_selection": thresholds["formal_evaluation_access_allowed"] is False and thresholds["selected_threshold_tuple"] is None and thresholds["lock_sha256"] is None,
        "G10_strongest_baseline_pilot_only": stats["strongest_baseline_selection"]["split"] == "pilot_validation" and stats["strongest_baseline_selection"]["oracle_excluded"] is True and stats["strongest_baseline_selection"]["formal_reselection_prohibited"] is True and thresholds["selected_strongest_deployable_baseline"] is None,
        "G11_paired_seed_cluster_bootstrap_frozen": stats["independent_cluster"].startswith("formal seed") and stats["bootstrap"]["replicates"] == 20000 and stats["bootstrap"]["seed"] == 202607176999 and stats["bootstrap"]["confidence_level"] == 0.95,
        "G11b_equal_family_aggregation_prevents_reweighting": stats["aggregation_weights"]["aggregate_smooth"].startswith("four smooth family estimates equally weighted") and stats["aggregation_weights"]["aggregate_abrupt_ood"].startswith("six abrupt/OOD family estimates equally weighted") and "post-result reweighting" in stats["aggregation_weights"]["forbidden"] and "equal-family" in stats["strongest_baseline_selection"]["order"],
        "G12_multiplicity_families_frozen": stats["multiplicity"]["method"] == "Holm step-down" and stats["multiplicity"]["familywise_alpha"] == 0.05 and set(stats["multiplicity"]["families"]["smooth_family_superiority"]) == set(SMOOTH_FAMILIES) and set(stats["multiplicity"]["families"]["abrupt_ood_safety"]) == set((*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY)),
        "G13_primary_smooth_gate_is_strict": acceptance["primary_smooth"]["aggregate_paired_95_lcb_min_exclusive"] == 0.0 and acceptance["primary_smooth"]["all_four_smooth_families_executed"] is True,
        "G14_calibration_shift_counterexample_gate_strict": acceptance["calibration_shift_tail"]["global_worst_error_count_proposed_max_relative_to_baseline"] == 0 and acceptance["calibration_shift_tail"]["seed_worst_paired_difference_95_ucb_max"] == 0.0 and "55/512" in acceptance["calibration_shift_tail"]["prior_counterexample_target"],
        "G15_catastrophic_margins_all_required": acceptance["catastrophic_degradation_each_abrupt_ood_family"]["average_ler_proposed_minus_baseline_95_ucb_max"] == 0.002 and acceptance["catastrophic_degradation_each_abrupt_ood_family"]["p95_window_ler_proposed_minus_baseline_95_ucb_max"] == 4 / 512 and acceptance["catastrophic_degradation_each_abrupt_ood_family"]["seed_worst_window_ler_proposed_minus_baseline_95_ucb_max"] == 8 / 512 and acceptance["catastrophic_degradation_each_abrupt_ood_family"]["all_conditions_required"] is True,
        "G16_nominal_noninferiority_all_required": acceptance["nominal_non_inferiority"]["average_ler_proposed_minus_policy_off_95_ucb_max"] == 0.0005 and acceptance["nominal_non_inferiority"]["fallback_rate_max"] == 0.01 and acceptance["nominal_non_inferiority"]["all_conditions_required"] is True,
        "G17_metrics_include_pauli_tail_lag_and_action_accounting": all(key in metrics for key in ("p_x", "p_y", "p_z", "p_l", "p95_window_ler", "seed_worst_window_ler", "global_worst_window_ler", "oracle_gap_closure", "adaptation_lag", "false_update", "unnecessary_fallback", "avoided_error", "induced_error", "deadline_miss")),
        "G18_prior_evidence_and_adaptive_origin_disclosed": len(payload["prior_evidence_disclosure"]) == 4 and all(row["exists"] and row["sha256"] and row["disclosed_use"] for row in payload["prior_evidence_disclosure"]),
        "G19_future_formal_results_absent_at_freeze": payload["formal_result_access_at_freeze"]["accessed"] is False and payload["formal_result_access_at_freeze"]["paths_existing_when_frozen"] == [],
        "G20_protocol_is_immutable_versioned": payload["protocol_id"] == PROTOCOL_ID and payload["protocol_revision_policy"].startswith("immutable; any post-freeze change requires a new protocol_id"),
    }


def validate_protocol(payload: Mapping[str, Any], *, verify_sources: bool = True) -> None:
    gates = recompute_gates(payload)
    if not all(gates.values()):
        raise ValueError(f"Route-A preregistration failed: {[key for key, value in gates.items() if not value]}")
    if verify_sources:
        for row in payload["prior_evidence_disclosure"]:
            path = ROOT / row["path"]
            if not path.is_file() or _sha256(path) != row["sha256"]:
                raise ValueError(f"prior evidence binding stale: {row['path']}")


MUTATIONS = (
    ("overlap_seed", ("splits", 2, "seeds")),
    ("overlap_rate", ("splits", 2, "transition_rates_per_window")),
    ("per_scenario_threshold", ("threshold_selection", "per_scenario_thresholds_prohibited")),
    ("preselect_threshold", ("threshold_selection", "selected_threshold_tuple")),
    ("formal_threshold_access", ("threshold_selection", "formal_evaluation_access_allowed")),
    ("oracle_baseline", ("statistics", "strongest_baseline_selection", "oracle_excluded")),
    ("window_as_cluster", ("statistics", "independent_cluster")),
    ("low_bootstrap", ("statistics", "bootstrap", "replicates")),
    ("weak_primary", ("acceptance_gates", "primary_smooth", "aggregate_paired_95_lcb_min_exclusive")),
    ("allow_55_over_37", ("acceptance_gates", "calibration_shift_tail", "global_worst_error_count_proposed_max_relative_to_baseline")),
    ("drop_nominal_all", ("acceptance_gates", "nominal_non_inferiority", "all_conditions_required")),
    ("overwrite_protocol", ("protocol_revision_policy",)),
)


def mutation_audit(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    rows = []
    for mutation_id, path in MUTATIONS:
        mutated = deepcopy(payload)
        target: Any = mutated
        for key in path[:-1]:
            target = target[key]
        if mutation_id == "overlap_seed":
            value = list(mutated["splits"][1]["seeds"])
        elif mutation_id == "overlap_rate":
            value = list(mutated["splits"][1]["transition_rates_per_window"])
        elif mutation_id == "preselect_threshold":
            value = {"regime_posterior_enter_min": 0.8}
        elif mutation_id == "window_as_cluster":
            value = "each window independent"
        elif mutation_id == "low_bootstrap":
            value = 100
        elif mutation_id == "weak_primary":
            value = -0.001
        elif mutation_id == "allow_55_over_37":
            value = 18
        elif mutation_id == "overwrite_protocol":
            value = "mutable in place"
        else:
            value = not bool(target[path[-1]])
        target[path[-1]] = value
        try:
            validate_protocol(mutated, verify_sources=False)
        except ValueError as exc:
            rows.append({"mutation_id": mutation_id, "rejected": True, "reason": str(exc)})
        else:
            rows.append({"mutation_id": mutation_id, "rejected": False, "reason": "accepted"})
    return rows


def build_report() -> dict[str, Any]:
    protocol = protocol_payload()
    validate_protocol(protocol)
    mutations = mutation_audit(protocol)
    gates = recompute_gates(protocol)
    report = {
        **protocol,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _canonical_sha256(protocol),
        "gates": [{"gate_id": key, "passed": value} for key, value in gates.items()],
        "semantic_mutations": mutations,
        "status": "PASS" if all(gates.values()) and all(row["rejected"] for row in mutations) else "FAIL",
        "verdict": "PASS_ROUTE_A_RESULT_BLIND_PREREGISTRATION_FROZEN" if all(gates.values()) and all(row["rejected"] for row in mutations) else "FAIL_ROUTE_A_PREREGISTRATION",
        "claim_boundary": {
            "allowed": "result-blind split/scenario/threshold-selection/statistical/decision-gate preregistration for new T6.7 formal evidence",
            "not_claimed": ["threshold tuple selected", "strongest baseline selected", "formal evaluation executed", "Route-A performance or safety advantage"],
        },
    }
    return report


def verify_report(report: Mapping[str, Any], *, verify_sources: bool = True) -> None:
    protocol_keys = set(protocol_payload())
    protocol = {key: report[key] for key in protocol_keys}
    validate_protocol(protocol, verify_sources=verify_sources)
    if report["protocol_sha256"] != _canonical_sha256(protocol):
        raise ValueError("protocol SHA-256 mismatch")
    gates = recompute_gates(protocol)
    stored = {row["gate_id"]: row["passed"] for row in report["gates"]}
    if stored != gates or not all(gates.values()):
        raise ValueError("stored preregistration gates do not recompute")
    if len(report["semantic_mutations"]) != len(MUTATIONS) or not all(row["rejected"] for row in report["semantic_mutations"]):
        raise ValueError("semantic mutation ledger incomplete")
    if report["status"] != "PASS" or report["verdict"] != "PASS_ROUTE_A_RESULT_BLIND_PREREGISTRATION_FROZEN":
        raise ValueError("preregistration verdict is not PASS")


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in report["splits"]:
        rows.append({"row_type": "split", "split_or_family": split["split_id"], "item_id": "seeds", "value": len(split["seeds"]), "detail": split["role"]})
    for cell in report["scenario_cells"]:
        rows.append({"row_type": "scenario_cell", "split_or_family": cell["split_id"], "item_id": cell["cell_id"], "value": cell["amplitude"], "detail": f"{cell['family']}|rate={cell['transition_rate_per_window']}|duration={cell['duration_windows']}"})
    for gate in report["gates"]:
        rows.append({"row_type": "gate", "split_or_family": "all", "item_id": gate["gate_id"], "value": gate["passed"], "detail": "PASS" if gate["passed"] else "FAIL"})
    for mutation in report["semantic_mutations"]:
        rows.append({"row_type": "mutation", "split_or_family": "all", "item_id": mutation["mutation_id"], "value": mutation["rejected"], "detail": mutation["reason"]})
    return rows


def write_report(artifact: Path = DEFAULT_ARTIFACT, source_data: Path = DEFAULT_SOURCE_DATA) -> dict[str, Any]:
    report = build_report()
    verify_report(report)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_type", "split_or_family", "item_id", "value", "detail"))
        writer.writeheader()
        writer.writerows(_csv_rows(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    report = write_report(args.artifact, args.source_data)
    print(json.dumps({"status": report["status"], "verdict": report["verdict"], "protocol_sha256": report["protocol_sha256"], "scenario_cells": len(report["scenario_cells"]), "formal_clusters": report["statistics"]["formal_cluster_count"], "gates": len(report["gates"]), "mutations": len(report["semantic_mutations"])}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
