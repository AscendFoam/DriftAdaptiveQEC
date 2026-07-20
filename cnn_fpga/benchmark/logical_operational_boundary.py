"""T5.3.3 full-curve simulated logical operational-boundary report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cnn_fpga.benchmark.logical_channel_fidelity import (
    DEFAULT_ARTIFACT as FIDELITY_ARTIFACT,
    implementation_sha256 as fidelity_implementation_sha256,
    validate_artifact_payload as validate_fidelity_payload,
)
from cnn_fpga.benchmark.logical_channel_reconstruction import (
    DEFAULT_ARTIFACT as CHANNEL_ARTIFACT,
    implementation_sha256 as channel_implementation_sha256,
    validate_artifact_payload as validate_channel_payload,
)
from physics.operational_boundary import matched_operational_boundary


TASK_ID = "T5.3.3"
CONTRACT_ID = "T533-FULL-CURVE-WALLCLOCK-OPERATIONAL-BOUNDARY-V1"
DEFAULT_ARTIFACT = Path("docs/t5_3_3_logical_operational_boundary.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_3_3_logical_operational_boundary_source_data.csv")
PAPER_SOURCE = Path(
    "relative_papers/Real-time_quantum_error_correction_beyond_break-even/"
    "Real-time_quantum_error_correction_beyond_break-even.md"
)
PAPER_FRAGMENTS = (
    "best passive qubit encoding",
    "We define the coherence gain G",
    "with the break-even point corresponding to G=1",
)
FORMAL_CUTOFFS = (12, 24, 36, 40)
TERMINAL_CUTOFFS = (36, 40)
NOISE_PROFILES = ("high", "medium", "low")


@dataclass(frozen=True)
class OperationalBoundaryConfig:
    channel_artifact: str = CHANNEL_ARTIFACT.as_posix()
    fidelity_artifact: str = FIDELITY_ARTIFACT.as_posix()
    formal_cutoffs: tuple[int, ...] = FORMAL_CUTOFFS
    terminal_cutoffs: tuple[int, int] = TERMINAL_CUTOFFS
    maximum_terminal_boundary_spread_us: float = 10.0

    def __post_init__(self) -> None:
        if Path(self.channel_artifact).as_posix() != CHANNEL_ARTIFACT.as_posix():
            raise ValueError("formal report must remain bound to the T5.3.1 channel artifact")
        if Path(self.fidelity_artifact).as_posix() != FIDELITY_ARTIFACT.as_posix():
            raise ValueError("formal report must remain bound to the T5.3.2 fidelity artifact")
        cutoffs = tuple(int(value) for value in self.formal_cutoffs)
        if cutoffs != FORMAL_CUTOFFS:
            raise ValueError("formal cutoff scan must remain 12/24/36/40")
        object.__setattr__(self, "formal_cutoffs", cutoffs)
        terminal = tuple(int(value) for value in self.terminal_cutoffs)
        if terminal != TERMINAL_CUTOFFS:
            raise ValueError("terminal cutoff repeat must remain 36/40")
        object.__setattr__(self, "terminal_cutoffs", terminal)
        if self.maximum_terminal_boundary_spread_us != 10.0:
            raise ValueError("formal terminal boundary tolerance must remain one 10 us sample")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "operational_boundary.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _deep_close(left: Any, right: Any, *, tolerance: float = 2.0e-9) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _deep_close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _deep_close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if left is None or right is None or isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
        return left == right
    try:
        return bool(np.isclose(float(left), float(right), rtol=2.0e-9, atol=tolerance))
    except (TypeError, ValueError):
        return left == right


def _load_parents() -> tuple[dict[str, Any], dict[str, Any]]:
    channel = json.loads(CHANNEL_ARTIFACT.read_text(encoding="utf-8"))
    fidelity = json.loads(FIDELITY_ARTIFACT.read_text(encoding="utf-8"))
    validate_channel_payload(channel)
    validate_fidelity_payload(fidelity)
    return channel, fidelity


def _parent_audits(channel: Mapping[str, Any], fidelity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "channel": {
            "path": CHANNEL_ARTIFACT.as_posix(),
            "sha256": _sha256(CHANNEL_ARTIFACT),
            "task_id": channel.get("task_id"),
            "status": channel.get("status"),
            "all_gates_passed": all(channel.get("gates", {}).values()),
            "stored_implementation_sha256": channel.get("implementation_sha256"),
            "live_implementation_sha256": channel_implementation_sha256(),
            "implementation_hash_matches": (
                channel.get("implementation_sha256") == channel_implementation_sha256()
            ),
        },
        "fidelity": {
            "path": FIDELITY_ARTIFACT.as_posix(),
            "sha256": _sha256(FIDELITY_ARTIFACT),
            "task_id": fidelity.get("task_id"),
            "status": fidelity.get("status"),
            "all_gates_passed": all(fidelity.get("gates", {}).values()),
            "stored_implementation_sha256": fidelity.get("implementation_sha256"),
            "live_implementation_sha256": fidelity_implementation_sha256(),
            "implementation_hash_matches": (
                fidelity.get("implementation_sha256") == fidelity_implementation_sha256()
            ),
            "channel_parent_sha256": fidelity.get("parent_audit", {}).get("sha256"),
        },
    }


def _paper_audit() -> dict[str, Any]:
    text = PAPER_SOURCE.read_text(encoding="utf-8")
    return {
        "path": PAPER_SOURCE.as_posix(),
        "sha256": _sha256(PAPER_SOURCE),
        "required_fragments": list(PAPER_FRAGMENTS),
        "fragment_matches": {fragment: fragment in text for fragment in PAPER_FRAGMENTS},
        "source_scope": (
            "the experimental coherence gain compares a qualified active rate to the best "
            "passive physical-qubit encoding; this project currently has only matched idle "
            "evolution of the same finite-cutoff encoded state"
        ),
    }


def _curves(fidelity_lane: Mapping[str, Any]) -> list[float]:
    return [float(point["average_fidelity"]) for point in fidelity_lane["cycle_metrics"]]


def _matching_audit(
    channel_on: Mapping[str, Any],
    channel_off: Mapping[str, Any],
    fidelity_on: Mapping[str, Any],
    fidelity_off: Mapping[str, Any],
) -> dict[str, Any]:
    on_config = dict(channel_on["config"])
    off_config = dict(channel_off["config"])
    on_mode = on_config.pop("mode")
    off_mode = off_config.pop("mode")
    return {
        "only_intervention_differs": (
            on_mode == "qec_on" and off_mode == "qec_off" and on_config == off_config
        ),
        "time_grid_matches": channel_on["time_us"] == channel_off["time_us"]
        == fidelity_on["time_us"] == fidelity_off["time_us"],
        "cycle_grid_matches": channel_on["cycles"] == channel_off["cycles"]
        == fidelity_on["cycles"] == fidelity_off["cycles"],
        "cycle_duration_us": on_config["cycle_duration_us"],
        "horizon_us": channel_on["time_us"][-1],
        "active_intervention": "fixed_nominal_sbs",
        "passive_intervention": "matched_idle_same_encoded_state",
        "baseline_role": "matched_uncorrected_grid_code_not_best_passive_physical_qubit",
        "wall_clock_matched": True,
        "active_control_operation_cost_measured": False,
        "full_cost_matched": False,
    }


def _derive_comparisons(
    channel: Mapping[str, Any], fidelity: Mapping[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for cutoff in FORMAL_CUTOFFS:
        for noise in NOISE_PROFILES:
            on_id = f"cutoff{cutoff}:{noise}:qec_on"
            off_id = f"cutoff{cutoff}:{noise}:qec_off"
            on = fidelity["lanes"][on_id]
            off = fidelity["lanes"][off_id]
            boundary = matched_operational_boundary(
                on["time_us"], _curves(on), _curves(off)
            ).to_dict()
            matching = _matching_audit(
                channel["lanes"][on_id], channel["lanes"][off_id], on, off
            )
            wall_clock_qualified = bool(
                matching["only_intervention_differs"]
                and matching["time_grid_matches"]
                and matching["cycle_grid_matches"]
                and boundary["initial_penalty_min"] < -2.0e-9
                and boundary["sustained_dominance_time_us"] is not None
                and boundary["cumulative_breakeven_time_us"] is not None
                and boundary["terminal_advantage"] > 2.0e-9
                and boundary["terminal_cumulative_advantage_us"] > 2.0e-9
            )
            comparisons[f"cutoff{cutoff}:{noise}"] = {
                "comparison_id": f"cutoff{cutoff}:{noise}",
                "cutoff": cutoff,
                "noise_profile": noise,
                "active_lane_id": on_id,
                "passive_lane_id": off_id,
                "matching_audit": matching,
                "boundary": boundary,
                "active_short_time_rate_status": on["short_time_effective_depolarization"][
                    "reliability_status"
                ],
                "active_qualified_short_time_lifetime_us": on[
                    "short_time_effective_depolarization"
                ]["primary_lifetime_us"],
                "passive_short_time_rate_status": off["short_time_effective_depolarization"][
                    "reliability_status"
                ],
                "wall_clock_boundary_qualified": wall_clock_qualified,
                "full_cost_boundary_qualified": False,
                "coherence_gain_qualified": False,
            }
    return comparisons


def _terminal_qualification(comparisons: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for noise in NOISE_PROFILES:
        low = comparisons[f"cutoff36:{noise}"]
        high = comparisons[f"cutoff40:{noise}"]
        low_boundary = low["boundary"]
        high_boundary = high["boundary"]
        sustained_spread = abs(
            low_boundary["sustained_dominance_time_us"]
            - high_boundary["sustained_dominance_time_us"]
        )
        cumulative_spread = abs(
            low_boundary["cumulative_breakeven_time_us"]
            - high_boundary["cumulative_breakeven_time_us"]
        )
        stable = bool(
            low["wall_clock_boundary_qualified"]
            and high["wall_clock_boundary_qualified"]
            and sustained_spread <= 10.0
            and cumulative_spread <= 10.0
        )
        rows.append(
            {
                "noise_profile": noise,
                "lower_cutoff": 36,
                "higher_cutoff": 40,
                "sustained_boundary_time_us_at_36": low_boundary[
                    "sustained_dominance_time_us"
                ],
                "sustained_boundary_time_us_at_40": high_boundary[
                    "sustained_dominance_time_us"
                ],
                "sustained_boundary_spread_us": sustained_spread,
                "cumulative_breakeven_time_us_at_36": low_boundary[
                    "cumulative_breakeven_time_us"
                ],
                "cumulative_breakeven_time_us_at_40": high_boundary[
                    "cumulative_breakeven_time_us"
                ],
                "cumulative_breakeven_spread_us": cumulative_spread,
                "terminal_repeat_stable": stable,
                "status": (
                    "QUALIFIED_WALL_CLOCK_OPERATIONAL_BOUNDARY"
                    if stable
                    else "NOT_QUALIFIED"
                ),
                "is_statistical_confidence_interval": False,
                "infinite_cutoff_claim": False,
            }
        )
    return rows


def _semantic_gates(
    payload: Mapping[str, Any],
    channel: Mapping[str, Any],
    fidelity: Mapping[str, Any],
) -> dict[str, bool]:
    comparisons = payload.get("comparisons", {})
    audits = payload.get("parent_audits", {})
    paper = payload.get("paper_audit", {})
    recomputed = _derive_comparisons(channel, fidelity)
    terminal = payload.get("terminal_qualification", [])
    verdict = payload.get("verdict", {})
    claim = payload.get("claim_boundary", {})
    low_cutoff = [comparisons.get(f"cutoff12:{noise}", {}) for noise in NOISE_PROFILES]
    terminal_comparisons = [
        comparisons.get(f"cutoff{cutoff}:{noise}", {})
        for cutoff in TERMINAL_CUTOFFS
        for noise in NOISE_PROFILES
    ]
    return {
        "t531_channel_parent_and_implementation_are_live": bool(
            audits.get("channel", {}).get("path") == CHANNEL_ARTIFACT.as_posix()
            and audits.get("channel", {}).get("sha256") == _sha256(CHANNEL_ARTIFACT)
            and audits.get("channel", {}).get("task_id") == "T5.3.1"
            and audits.get("channel", {}).get("status") == "PASS"
            and audits.get("channel", {}).get("all_gates_passed")
            and audits.get("channel", {}).get("implementation_hash_matches")
        ),
        "t532_fidelity_parent_and_implementation_are_live": bool(
            audits.get("fidelity", {}).get("path") == FIDELITY_ARTIFACT.as_posix()
            and audits.get("fidelity", {}).get("sha256") == _sha256(FIDELITY_ARTIFACT)
            and audits.get("fidelity", {}).get("task_id") == "T5.3.2"
            and audits.get("fidelity", {}).get("status") == "PASS"
            and audits.get("fidelity", {}).get("all_gates_passed")
            and audits.get("fidelity", {}).get("implementation_hash_matches")
        ),
        "fidelity_parent_remains_bound_to_live_channel_artifact": (
            audits.get("fidelity", {}).get("channel_parent_sha256")
            == _sha256(CHANNEL_ARTIFACT)
        ),
        "primary_paper_best_passive_and_gain_definition_are_anchored": bool(
            paper.get("path") == PAPER_SOURCE.as_posix()
            and paper.get("sha256") == _sha256(PAPER_SOURCE)
            and all(paper.get("fragment_matches", {}).values())
        ),
        "all_12_cutoff_noise_comparisons_are_present": (
            len(comparisons) == 12
            and set(comparisons) == {
                f"cutoff{cutoff}:{noise}"
                for cutoff in FORMAL_CUTOFFS
                for noise in NOISE_PROFILES
            }
        ),
        "active_and_passive_share_model_time_grid_and_horizon": all(
            comparison.get("matching_audit", {}).get("only_intervention_differs")
            and comparison.get("matching_audit", {}).get("time_grid_matches")
            and comparison.get("matching_audit", {}).get("cycle_grid_matches")
            and comparison.get("matching_audit", {}).get("cycle_duration_us") == 10.0
            and comparison.get("matching_audit", {}).get("horizon_us") == 300.0
            for comparison in comparisons.values()
        ) if comparisons else False,
        "baseline_is_matched_uncorrected_grid_not_best_physical_qubit": all(
            comparison.get("matching_audit", {}).get("baseline_role")
            == "matched_uncorrected_grid_code_not_best_passive_physical_qubit"
            for comparison in comparisons.values()
        ) if comparisons else False,
        "full_31_point_curves_are_preserved": all(
            len(comparison.get("boundary", {}).get("time_us", [])) == 31
            and len(comparison.get("boundary", {}).get("pointwise_advantage", [])) == 31
            and len(comparison.get("boundary", {}).get("cumulative_advantage_us", [])) == 31
            for comparison in comparisons.values()
        ) if comparisons else False,
        "sustained_boundary_requires_all_later_samples_noninferior": all(
            comparison["boundary"]["sustained_dominance_index"] is None
            or all(
                value >= -2.0e-9
                for value in comparison["boundary"]["pointwise_advantage"][
                    comparison["boundary"]["sustained_dominance_index"] :
                ]
            )
            for comparison in comparisons.values()
        ) if comparisons else False,
        "cumulative_boundary_repays_initial_deficit_and_stays_nonnegative": all(
            comparison["boundary"]["cumulative_breakeven_index"] is None
            or all(
                value >= -2.0e-9
                for value in comparison["boundary"]["cumulative_advantage_us"][
                    comparison["boundary"]["cumulative_breakeven_index"] :
                ]
            )
            for comparison in comparisons.values()
        ) if comparisons else False,
        "initial_active_penalty_and_sign_reversals_are_not_hidden": all(
            comparison["boundary"]["initial_penalty_min"] < -0.01
            and comparison["boundary"]["sign_reversal_count"] >= 1
            for comparison in terminal_comparisons
        ) if terminal_comparisons else False,
        "terminal_cutoffs_establish_sustained_and_cumulative_boundaries": all(
            comparison.get("wall_clock_boundary_qualified") is True
            and comparison["boundary"]["sustained_dominance_time_us"] is not None
            and comparison["boundary"]["cumulative_breakeven_time_us"] is not None
            for comparison in terminal_comparisons
        ) if terminal_comparisons else False,
        "terminal_36_40_boundary_repeat_is_stable_within_one_sample": (
            len(terminal) == 3
            and all(
                row.get("terminal_repeat_stable") is True
                and row.get("sustained_boundary_spread_us") <= 10.0
                and row.get("cumulative_breakeven_spread_us") <= 10.0
                and row.get("is_statistical_confidence_interval") is False
                and row.get("infinite_cutoff_claim") is False
                for row in terminal
            )
        ),
        "cutoff12_failure_is_retained_not_selected_away": all(
            comparison.get("wall_clock_boundary_qualified") is False
            and comparison.get("boundary", {}).get("sustained_dominance_time_us") is None
            and comparison.get("boundary", {}).get("cumulative_breakeven_time_us") is None
            and comparison.get("boundary", {}).get("terminal_advantage") < 0.0
            for comparison in low_cutoff
        ) if low_cutoff else False,
        "active_short_time_rate_failure_is_carried_forward": all(
            comparison.get("active_short_time_rate_status")
            == "unreliable_cycle_scale_transient"
            and comparison.get("active_qualified_short_time_lifetime_us") is None
            for comparison in comparisons.values()
        ) if comparisons else False,
        "wall_clock_is_matched_but_active_control_cost_is_not_faked": all(
            comparison.get("matching_audit", {}).get("wall_clock_matched") is True
            and comparison.get("matching_audit", {}).get(
                "active_control_operation_cost_measured"
            ) is False
            and comparison.get("matching_audit", {}).get("full_cost_matched") is False
            and comparison.get("full_cost_boundary_qualified") is False
            for comparison in comparisons.values()
        ) if comparisons else False,
        "no_ratio_or_exponential_substitute_is_reported": all(
            comparison.get("boundary", {}).get("ratio_reported") is False
            and comparison.get("boundary", {}).get("exponential_fit_used") is False
            for comparison in comparisons.values()
        ) if comparisons else False,
        "linear_interpolation_is_diagnostic_not_subgrid_validation": all(
            comparison.get("boundary", {}).get("interpolation_role")
            == "linear_between_sample_diagnostic_not_subgrid_validation"
            for comparison in comparisons.values()
        ) if comparisons else False,
        "all_comparisons_recompute_from_parent_full_curves": _deep_close(
            recomputed, comparisons
        ),
        "terminal_table_recomputes_from_comparisons": _deep_close(
            _terminal_qualification(comparisons), terminal
        ) if comparisons else False,
        "wall_clock_boundary_verdict_is_qualified_and_finite_horizon": bool(
            verdict.get("wall_clock_operational_boundary")
            == "ESTABLISHED_WITHIN_300US_FINITE_CUTOFF_MODEL"
            and verdict.get("terminal_cutoff_profiles_qualified") == 3
            and verdict.get("low_cutoff_counterexample_retained") is True
        ),
        "full_cost_boundary_remains_pending_t534": (
            verdict.get("full_cost_operational_boundary")
            == "NOT_ESTABLISHED_PENDING_T5.3.4"
        ),
        "paper_definition_coherence_gain_remains_not_established": bool(
            verdict.get("simulation_derived_coherence_gain") == "NOT_ESTABLISHED"
            and verdict.get("coherence_gain_value") is None
            and all(
                comparison.get("coherence_gain_qualified") is False
                for comparison in comparisons.values()
            )
        ) if comparisons else False,
        "statistical_uncertainty_is_not_invented_for_exact_curves": bool(
            payload.get("uncertainty_contract", {}).get("statistical_standard_error") is None
            and payload.get("uncertainty_contract", {}).get("statistical_confidence_interval") is None
            and payload.get("uncertainty_contract", {}).get("terminal_repeat_is_ci") is False
        ),
        "claim_boundary_remains_simulation_only_without_break_even_promotion": bool(
            claim.get("experimental_break_even") is False
            and claim.get("physical_memory_break_even") is False
            and claim.get("best_passive_physical_reference_present") is False
            and claim.get("target_hardware_measured") is False
        ),
    }


def validate_artifact_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    if payload.get("task_id") != TASK_ID or payload.get("contract_id") != CONTRACT_ID:
        raise ValueError("artifact task/contract identity mismatch")
    channel, fidelity = _load_parents()
    gates = _semantic_gates(payload, channel, fidelity)
    if payload.get("gates") != gates:
        raise ValueError("stored gates do not match recomputed semantic gates")
    if payload.get("required_gates") != list(gates):
        raise ValueError("required gate order/schema mismatch")
    expected = "PASS" if all(gates.values()) else "FAIL"
    if payload.get("status") != expected:
        raise ValueError("artifact status does not match gates")
    return gates


def run_report(config: OperationalBoundaryConfig = OperationalBoundaryConfig()) -> dict[str, Any]:
    channel, fidelity = _load_parents()
    comparisons = _derive_comparisons(channel, fidelity)
    terminal = _terminal_qualification(comparisons)
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "status": "PENDING",
        "implementation_sha256": implementation_sha256(),
        "config": {
            "channel_artifact": config.channel_artifact,
            "fidelity_artifact": config.fidelity_artifact,
            "formal_cutoffs": list(config.formal_cutoffs),
            "terminal_cutoffs": list(config.terminal_cutoffs),
            "maximum_terminal_boundary_spread_us": config.maximum_terminal_boundary_spread_us,
        },
        "parent_audits": _parent_audits(channel, fidelity),
        "paper_audit": _paper_audit(),
        "boundary_contract": {
            "primary_metric": "leakage-inclusive CPTNI average fidelity",
            "sampled_sustained_boundary": (
                "first 10 us sampled time after which active-minus-passive F_avg remains "
                "nonnegative through the 300 us horizon"
            ),
            "cumulative_breakeven": (
                "first sampled time after which trapezoidal integral of active-minus-passive "
                "F_avg remains nonnegative through the 300 us horizon"
            ),
            "baseline": "matched idle evolution of the same encoded finite-cutoff state",
            "cost_scope": "same model and wall-clock grid; active operation cost remains unpriced",
            "fit": None,
            "ratio": None,
        },
        "uncertainty_contract": {
            "statistical_standard_error": None,
            "statistical_confidence_interval": None,
            "reason": "deterministic exact full-curve comparison has no stochastic sample population",
            "terminal_repeat": "36/40 sampled-boundary sensitivity",
            "terminal_repeat_is_ci": False,
            "subgrid_interpolation_is_validation": False,
        },
        "comparisons": comparisons,
        "terminal_qualification": terminal,
        "verdict": {
            "wall_clock_operational_boundary": (
                "ESTABLISHED_WITHIN_300US_FINITE_CUTOFF_MODEL"
                if all(row["terminal_repeat_stable"] for row in terminal)
                else "NOT_ESTABLISHED"
            ),
            "terminal_cutoff_profiles_qualified": sum(
                row["terminal_repeat_stable"] for row in terminal
            ),
            "low_cutoff_counterexample_retained": all(
                not comparisons[f"cutoff12:{noise}"]["wall_clock_boundary_qualified"]
                for noise in NOISE_PROFILES
            ),
            "full_cost_operational_boundary": "NOT_ESTABLISHED_PENDING_T5.3.4",
            "simulation_derived_coherence_gain": "NOT_ESTABLISHED",
            "coherence_gain_value": None,
            "coherence_gain_blockers": [
                "active short-time rate is unreliable because of cycle-scale transient",
                "matched qec_off is not the best passive physical-qubit encoding",
                "active operation and reset cost is not yet fully priced",
            ],
            "experimental_break_even": "NOT_ESTABLISHED",
        },
        "claim_boundary": {
            "allowed": (
                "simulation-derived wall-clock operational boundary for matched finite-cutoff "
                "active nominal-SBS versus uncorrected encoded idle curves"
            ),
            "forbidden": [
                "coherence gain under the paper best-passive-rate definition",
                "single-point or terminal-only break-even",
                "raw area ratio or exponential lifetime substitute",
                "full-cost operational boundary before T5.3.4",
                "infinite-cutoff, physical-memory or experimental break-even",
                "device, QPU or FPGA performance claim",
            ],
            "experimental_break_even": False,
            "physical_memory_break_even": False,
            "best_passive_physical_reference_present": False,
            "target_hardware_measured": False,
        },
    }
    gates = _semantic_gates(payload, channel, fidelity)
    payload["gates"] = gates
    payload["required_gates"] = list(gates)
    payload["status"] = "PASS" if all(gates.values()) else "FAIL"
    validate_artifact_payload(payload)
    return payload


def _source_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "category": "contract",
            "task_id": TASK_ID,
            "contract_id": CONTRACT_ID,
            "status": payload["status"],
            "implementation_sha256": payload["implementation_sha256"],
        },
        {"category": "paper", "path": payload["paper_audit"]["path"], "sha256": payload["paper_audit"]["sha256"]},
    ]
    for name, audit in payload["parent_audits"].items():
        rows.append({"category": "parent", "parent": name, **audit})
    for comparison in payload["comparisons"].values():
        boundary = comparison["boundary"]
        for index, time_us in enumerate(boundary["time_us"]):
            rows.append(
                {
                    "category": "full_curve",
                    "comparison_id": comparison["comparison_id"],
                    "cutoff": comparison["cutoff"],
                    "noise_profile": comparison["noise_profile"],
                    "cycle": index,
                    "time_us": time_us,
                    "active_fidelity": boundary["active_fidelity"][index],
                    "passive_fidelity": boundary["passive_fidelity"][index],
                    "pointwise_advantage": boundary["pointwise_advantage"][index],
                    "cumulative_advantage_us": boundary["cumulative_advantage_us"][index],
                }
            )
        rows.append(
            {
                "category": "boundary_summary",
                "comparison_id": comparison["comparison_id"],
                "cutoff": comparison["cutoff"],
                "noise_profile": comparison["noise_profile"],
                **{
                    key: value
                    for key, value in boundary.items()
                    if key
                    not in {
                        "time_us",
                        "active_fidelity",
                        "passive_fidelity",
                        "pointwise_advantage",
                        "cumulative_advantage_us",
                    }
                },
                **comparison["matching_audit"],
                "wall_clock_boundary_qualified": comparison[
                    "wall_clock_boundary_qualified"
                ],
                "full_cost_boundary_qualified": comparison["full_cost_boundary_qualified"],
                "coherence_gain_qualified": comparison["coherence_gain_qualified"],
            }
        )
    rows.extend(
        {"category": "terminal_qualification", **row}
        for row in payload["terminal_qualification"]
    )
    rows.extend(
        {"category": "gate", "gate": gate, "passed": passed}
        for gate, passed in payload["gates"].items()
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    config: OperationalBoundaryConfig = OperationalBoundaryConfig(),
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = run_report(config)
    rows = _source_rows(payload)
    _write_csv(source_data_path, rows)
    payload["source_data"] = {
        "path": source_data_path.as_posix(),
        "sha256": _sha256(source_data_path),
        "row_count": len(rows),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args()
    payload = write_artifacts(artifact_path=args.artifact, source_data_path=args.source_data)
    print(
        json.dumps(
            {
                "task_id": payload["task_id"],
                "status": payload["status"],
                "gates": f"{sum(payload['gates'].values())}/{len(payload['gates'])}",
                "source_rows": payload["source_data"]["row_count"],
                "verdict": payload["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "OperationalBoundaryConfig",
    "implementation_sha256",
    "run_report",
    "validate_artifact_payload",
    "write_artifacts",
]

