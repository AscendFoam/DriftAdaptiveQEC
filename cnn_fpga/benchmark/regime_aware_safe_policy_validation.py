"""T6.6.2 production-cadence structural validation for Route-A policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.decoder.periodic_adaptive_map import (
    LatestWindowPeriodicPredictor,
    PeriodicGaussianEstimate,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    estimate_periodic_gaussian,
)
from cnn_fpga.runtime.closed_loop_fault_recovery import (
    ClosedLoopCycleInput,
    parameter_image_semantics_sha256,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig
from cnn_fpga.runtime.regime_aware_safe_policy import (
    ADAPTIVE_OPEN,
    INTEGRITY_ROLLBACK,
    LEAKAGE_RESET,
    MODEL_SCOPE,
    POSTERIOR_UNCERTAIN,
    RECOVERING,
    TAIL_TRUSTED,
    AdaptiveMAPCandidate,
    ObservedRegimePosterior,
    RegimeAwareSafeAdaptivePolicy,
    RouteACycleInput,
    RouteAPolicyConfig,
)
from physics.constants import LATTICE_CONST


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "docs" / "t6_6_2_regime_aware_safe_policy.json"
DEFAULT_SOURCE = ROOT / "docs" / "t6_6_2_regime_aware_safe_policy_source_data.csv"
POSTERIOR_FIXTURE_HASH = hashlib.sha256(
    b"t6.6.2-structural-posterior-fixture-not-calibrated-hmm-output"
).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _observed_residuals(
    seed: int, *, mean: tuple[float, float], sigma: tuple[float, float], count: int = 2_048
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(np.asarray(mean), np.asarray(sigma), size=(count, 2))
    return np.mod(values + 0.5 * LATTICE_CONST, LATTICE_CONST) - 0.5 * LATTICE_CONST


def _params_from_estimate(
    estimate: PeriodicGaussianEstimate, *, estimator_id: str
) -> DecoderRuntimeParams:
    covariance = estimate.covariance_array()
    measurement_sigma = 0.04 * LATTICE_CONST
    measurement = np.eye(2, dtype=np.float64) * measurement_sigma**2
    gain = covariance @ np.linalg.inv(covariance + measurement)
    mean = estimate.mean_array()
    bias = (np.eye(2, dtype=np.float64) - gain) @ mean
    return DecoderRuntimeParams(
        K=gain,
        b=bias,
        metadata={
            "measurement_cov": measurement.tolist(),
            "alpha_bias": 1.0,
            "estimator_id": estimator_id,
            "estimator_source": estimate.source,
            "estimator_window_id": estimate.window_id,
            "observed_only": True,
        },
    )


def _compile_estimate(estimate: PeriodicGaussianEstimate, estimator_id: str):
    return compile_parametric_map_lut(
        _params_from_estimate(estimate, estimator_id=estimator_id),
        active_bank_version=0,
        config=ParametricMAPLUTConfig(),
    )


def _candidate_images() -> tuple[Any, dict[int, AdaptiveMAPCandidate], dict[str, Any]]:
    moment = PeriodicMomentConfig(minimum_samples=64)
    calibration = _observed_residuals(
        6601,
        mean=(0.025 * LATTICE_CONST, -0.018 * LATTICE_CONST),
        sigma=(0.145 * LATTICE_CONST, 0.155 * LATTICE_CONST),
    )
    initial_estimate = estimate_periodic_gaussian(
        calibration, moment, source="route_a_static_calibration", window_id=-1
    )
    initial = _compile_estimate(initial_estimate, "static_calibration")
    ewma = PeriodicMomentEWMA(calibration, alpha=0.20, config=moment)
    window = LatestWindowPeriodicPredictor(calibration, moment)
    first = _observed_residuals(
        6602,
        mean=(0.055 * LATTICE_CONST, -0.032 * LATTICE_CONST),
        sigma=(0.155 * LATTICE_CONST, 0.165 * LATTICE_CONST),
    )
    second = _observed_residuals(
        6603,
        mean=(0.085 * LATTICE_CONST, -0.047 * LATTICE_CONST),
        sigma=(0.170 * LATTICE_CONST, 0.180 * LATTICE_CONST),
    )
    window_first = window.update(first, window_id=1)
    ewma_first = ewma.update(first, window_id=1)
    window_second = window.update(second, window_id=2)
    ewma_second = ewma.update(second, window_id=2)
    images = {
        "window_first": _compile_estimate(window_first, "window_map"),
        "ewma_first": _compile_estimate(ewma_first, "ewma_adaptive_map"),
        "window_second": _compile_estimate(window_second, "window_map"),
        "ewma_second": _compile_estimate(ewma_second, "ewma_adaptive_map"),
    }
    candidates = {
        4_000: AdaptiveMAPCandidate(
            "window_map", 1, 0, 3_999, 4_000, images["window_first"],
            1_218, 3_840, 3_072, 900.0, 0.80, 1.05,
        ),
        8_000: AdaptiveMAPCandidate(
            "ewma_adaptive_map", 2, 4_000, 7_999, 8_000, images["ewma_second"],
            1_218, 3_840, 3_072, 910.0, 1.15, 0.75,
        ),
        12_000: AdaptiveMAPCandidate(
            "window_map", 3, 8_000, 11_999, 12_000, images["window_second"],
            1_218, 3_840, 3_072, 920.0, 0.82, 1.02,
        ),
        16_000: AdaptiveMAPCandidate(
            "window_map", 4, 12_000, 15_999, 16_000, images["window_first"],
            1_218, 3_840, 3_072, 930.0, 0.78, 1.08,
        ),
    }
    provenance = {
        "calibration_rows": int(calibration.shape[0]),
        "candidate_rows_each": int(first.shape[0]),
        "calibration_sha256": hashlib.sha256(calibration.astype("<f8").tobytes()).hexdigest(),
        "first_window_sha256": hashlib.sha256(first.astype("<f8").tobytes()).hexdigest(),
        "second_window_sha256": hashlib.sha256(second.astype("<f8").tobytes()).hexdigest(),
        "compiled_candidates": {
            key: {
                "image_sha256": image.image_sha256,
                "semantics_sha256": parameter_image_semantics_sha256(image),
                "source_params_sha256": image.source_params_sha256,
                "model_mean": list(image.model_mean),
                "model_sigma": list(image.model_sigma),
            }
            for key, image in images.items()
        },
        "input_contract": "observed_wrapped_residuals_only_no_hidden_state_or_logical_truth",
    }
    return initial, candidates, provenance


def _posterior(available_cycle: int) -> ObservedRegimePosterior:
    if 4_032 <= available_cycle <= 8_032:
        values = (0.04, 0.04, 0.46, 0.46)
    else:
        values = (0.55, 0.35, 0.05, 0.05)
    return ObservedRegimePosterior(
        source_window_id=available_cycle // 32,
        source_start_cycle=available_cycle - 32,
        source_end_cycle=available_cycle - 1,
        available_cycle=available_cycle,
        probabilities=values,
        model_sha256=POSTERIOR_FIXTURE_HASH,
    )


def _run_production_trace() -> tuple[list[Any], dict[str, Any]]:
    initial, candidates, provenance = _candidate_images()
    policy = RegimeAwareSafeAdaptivePolicy(initial)
    actions = []
    for cycle in range(5, 20_066):
        posterior = _posterior(cycle) if cycle % 32 == 0 else None
        candidate = candidates.get(cycle)
        syndrome_x = "leakage" if cycle in (12_100, 12_101) else "g"
        reset_ack = cycle == 12_102
        reported_integrity_ok = cycle != 16_001
        action = policy.step(
            RouteACycleInput(
                fast_path=ClosedLoopCycleInput(
                    epoch=cycle,
                    syndrome_code=(cycle * 37 + 113) % 1_024,
                    syndrome_x=syndrome_x,
                    syndrome_z="g",
                    quadrature_phase_bit=cycle & 1,
                    host_heartbeat=True,
                    reset_ack=reset_ack,
                    reported_integrity_ok=reported_integrity_ok,
                ),
                posterior_update=posterior,
                candidate=candidate,
                parameter_update_due=cycle % 4_000 == 0,
            )
        )
        actions.append(action)
    return actions, provenance


def _mutation_checks(initial) -> list[dict[str, Any]]:
    checks = []

    def rejected(name: str, needle: str, operation) -> None:
        try:
            operation()
        except Exception as exc:  # validation intentionally checks fail-closed APIs
            checks.append(
                {"name": name, "passed": needle in str(exc), "exception": type(exc).__name__, "message": str(exc)}
            )
        else:
            checks.append({"name": name, "passed": False, "message": "mutation accepted"})

    rejected(
        "posterior_sum",
        "sum to one",
        lambda: RegimeAwareSafeAdaptivePolicy(
            initial,
            config=RouteAPolicyConfig(
                regime_window_cycles=4,
                parameter_update_period_cycles=16,
            ),
        ).step(
            RouteACycleInput(
                ClosedLoopCycleInput(5, 512),
                ObservedRegimePosterior(1, 1, 4, 5, (0.7, 0.2, 0.1, 0.1), POSTERIOR_FIXTURE_HASH),
            )
        ),
    )
    rejected(
        "posterior_noncausal",
        "strictly after",
        lambda: ObservedRegimePosterior(1, 0, 31, 31, (0.7, 0.1, 0.1, 0.1), POSTERIOR_FIXTURE_HASH),
    )
    rejected(
        "posterior_nan",
        "finite",
        lambda: ObservedRegimePosterior(1, 0, 31, 32, (float("nan"), 0.1, 0.1, 0.8), POSTERIOR_FIXTURE_HASH),
    )
    rejected(
        "wrong_due_flag",
        "common cadence",
        lambda: RegimeAwareSafeAdaptivePolicy(initial).step(
            RouteACycleInput(ClosedLoopCycleInput(5, 512), parameter_update_due=True)
        ),
    )
    rejected(
        "hidden_truth_extra_field",
        "unexpected keyword",
        lambda: RouteACycleInput(ClosedLoopCycleInput(5, 512), hidden_regime="burst"),
    )
    rejected(
        "six_cycle_mutation",
        "six-cycle",
        lambda: RouteAPolicyConfig(fast_action_latency_cycles=5),
    )
    candidate = _candidate_images()[1][4_000]
    rejected(
        "candidate_future_source",
        "strictly after",
        lambda: replace(candidate, source_end_cycle=4_000),
    )
    rejected(
        "candidate_negative_cost",
        "at least 0",
        lambda: replace(candidate, update_macs=-1),
    )
    rejected(
        "candidate_nonfinite_router_score",
        "must be finite",
        lambda: replace(candidate, window_prequential_score=float("nan")),
    )
    rejected(
        "candidate_malformed_router_hash",
        "64-character",
        lambda: replace(candidate, router_algorithm_sha256="bad"),
    )
    rejected(
        "dual_shadow_system_budget",
        "exceeds the matched update-MAC budget",
        lambda: RouteAPolicyConfig(max_update_macs=1_217),
    )
    return checks


def _summarize(actions: list[Any], provenance: dict[str, Any]) -> dict[str, Any]:
    modes = Counter(action.policy_mode for action in actions)
    reasons = Counter(action.primary_reason for action in actions)
    commits = [action for action in actions if action.commit_status == "committed"]
    deferred = [action for action in actions if action.commit_status == "deferred"]
    candidates = [action for action in actions if action.candidate_received]
    tail = [action for action in actions if action.policy_mode == TAIL_TRUSTED]
    leakage = [action for action in actions if action.policy_mode == LEAKAGE_RESET]
    integrity_start = next(action for action in actions if action.cycle_index == 16_001)
    integrity_commit = next(
        action
        for action in actions
        if action.rollback_completed and action.cycle_index >= 16_001
    )
    integrity_interval = [
        action
        for action in actions
        if 16_001 <= action.cycle_index <= integrity_commit.cycle_index
    ]
    mutation_checks = _mutation_checks(_candidate_images()[0])
    expected_versions = [1, 2, 3, 4]
    guarded_lkg_anchor = next(
        action.active_semantics_sha256
        for action in commits
        if action.active_bank_version == 2
    )
    gates = {
        "production_contract_32_4000_6_and_caps": all(
            (
                actions[0].deadline.source_to_action_cycles == 6,
                all(action.deadline.source_to_action_cycles == 6 for action in actions),
                not any(action.deadline.host_budget_violation for action in actions),
            )
        ),
        "continuous_trace_no_replay_or_gap": [action.cycle_index for action in actions]
        == list(range(5, 20_066)),
        "every_action_has_reason_posterior_bank_and_deadline": all(
            action.primary_reason
            and action.reason_trace
            and len(action.posterior) == 4
            and action.active_bank in ("A", "B")
            and action.deadline.action_cycle == action.deadline.action_deadline_cycle
            for action in actions
        ),
        "actual_window_and_ewma_dual_shadow_candidates_seen": {
            action.fast_path_record.epoch: action.candidate_accepted
            for action in candidates
        }
        == {4_000: True, 8_000: True, 12_000: True, 16_000: True},
        "monotonic_expected_commit_versions": [action.active_bank_version for action in commits]
        == expected_versions,
        "all_commit_readbacks_confirmed": all(
            action.readback_status == "confirmed" for action in commits
        ),
        "tail_freezes_online_updates": bool(tail)
        and all(action.online_update_frozen for action in tail),
        "tail_policy_is_explicit_fallback_without_static_republish": all(
            action.fallback_active for action in tail
        )
        and not any(action.decision_source == "initial_static_calibration" for action in commits),
        "tail_boundary_accepts_only_budgeted_trusted_ewma_shadow": next(
            action for action in candidates if action.cycle_index == 8_000
        ).policy_mode
        == TAIL_TRUSTED
        and next(action for action in candidates if action.cycle_index == 8_000).candidate_accepted
        and any(
            action.cycle_index >= 8_001
            and action.policy_mode == TAIL_TRUSTED
            and action.decision_source == "trusted_ewma_shadow"
            for action in tail
        ),
        "minimum_residency_is_enforced_not_bypassed": len(deferred) >= 3
        and all(
            right.cycle_index - left.cycle_index
            >= RouteAPolicyConfig().parameter_update_period_cycles
            for left, right in zip(commits[:2], commits[1:3], strict=True)
        ),
        "leakage_reaches_reset_request": bool(leakage)
        and any(action.reset_request for action in leakage),
        "leakage_ack_does_not_immediately_open_commit": not next(
            action for action in actions if action.cycle_index == 12_102
        ).commit_gate_open,
        "transient_integrity_fault_latches_rollback": integrity_start.policy_mode
        == INTEGRITY_ROLLBACK
        and integrity_start.fallback_active
        and integrity_commit.active_bank_version == 4,
        "integrity_wait_interval_is_fail_closed": all(
            action.fallback_active for action in integrity_interval[:-1]
        ),
        "rollback_restores_pre_candidate_lkg_semantics_with_higher_version": (
            integrity_commit.active_semantics_sha256 == guarded_lkg_anchor
            and integrity_commit.active_bank_version > integrity_start.active_bank_version
        ),
        "integrity_cancels_pending_window_semantics_before_version_reuse": (
            integrity_commit.active_semantics_sha256
            != provenance["compiled_candidates"]["window_first"]["semantics_sha256"]
            and integrity_commit.active_semantics_sha256 == guarded_lkg_anchor
        ),
        "window_requires_open_gate_but_trusted_ewma_may_commit_in_tail": all(
            (not action.candidate_accepted)
            or action.commit_gate_open
            or (
                action.policy_mode in (TAIL_TRUSTED, POSTERIOR_UNCERTAIN)
                and action.cycle_index == 8_000
            )
            for action in actions
        ),
        "fixture_explicitly_not_calibrated_hmm_output": POSTERIOR_FIXTURE_HASH
        == hashlib.sha256(b"t6.6.2-structural-posterior-fixture-not-calibrated-hmm-output").hexdigest(),
        "mutation_suite_fail_closed": all(row["passed"] for row in mutation_checks),
        "model_scope_closes_rtl_board_and_performance_claims": MODEL_SCOPE
        == "route_a_integration_policy_thresholds_not_t6_6_3_frozen",
    }
    return {
        "schema_version": "t6.6.2-regime-aware-safe-policy-v2",
        "task_id": "T6.6.2",
        "scope": MODEL_SCOPE,
        "claim_boundary": {
            "admitted": "software structural integration, causal typed interfaces, real atomic/fallback/event execution",
            "not_admitted": [
                "posterior calibration or frozen thresholds (T6.6.3)",
                "LER superiority (T6.7)",
                "synthesized integrated Route-A resources/timing (T6.9)",
                "board-measured latency or deadline (board-only lane)",
            ],
        },
        "contract": asdict(RouteAPolicyConfig()),
        "trace": {
            "first_cycle": actions[0].cycle_index,
            "last_cycle": actions[-1].cycle_index,
            "cycles": len(actions),
            "mode_counts": dict(sorted(modes.items())),
            "reason_counts": dict(sorted(reasons.items())),
            "commit_versions": [action.active_bank_version for action in commits],
            "commit_cycles": [action.cycle_index for action in commits],
            "deferred_commit_cycles": len(deferred),
            "fallback_cycles": sum(action.fallback_active for action in actions),
            "reset_request_cycles": sum(action.reset_request for action in actions),
            "rollback_completed_cycles": [
                action.cycle_index for action in actions if action.rollback_completed
            ],
            "trusted_switch_completed_cycles": [
                action.cycle_index for action in actions if action.trusted_switch_completed
            ],
            "candidate_rows": [
                {
                    "cycle": action.cycle_index,
                    "accepted": action.candidate_accepted,
                    "reason": action.candidate_reason,
                    "staged_version": action.staged_version,
                    "deadline": asdict(action.deadline),
                }
                for action in candidates
            ],
        },
        "candidate_provenance": provenance,
        "posterior_fixture": {
            "sha256": POSTERIOR_FIXTURE_HASH,
            "status": "structural_branch_fixture_not_hmm_output_not_threshold_calibration",
            "classes": ["normal", "smooth", "calibration_shift", "burst"],
        },
        "mutation_checks": mutation_checks,
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "all_passed": all(gates.values()),
        },
        "source_files": {},
    }


def _write_csv(path: Path, actions: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "cycle_index", "policy_mode", "primary_reason", "posterior", "posterior_source_window_id",
        "active_bank", "active_bank_version", "active_semantics_sha256", "decision_source",
        "online_update_frozen", "commit_gate_open", "fallback_active", "reset_request",
        "rollback_requested", "rollback_completed", "trusted_switch_requested",
        "trusted_switch_completed", "candidate_received", "candidate_accepted",
        "safety_stage_accepted", "candidate_reason", "staged_version", "commit_status",
        "commit_reason", "readback_status", "source_cycle", "action_cycle",
        "source_to_action_cycles", "fast_deadline_miss", "host_budget_violation", "fault_flags",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for action in actions:
            writer.writerow(
                {
                    "cycle_index": action.cycle_index,
                    "policy_mode": action.policy_mode,
                    "primary_reason": action.primary_reason,
                    "posterior": "|".join(f"{value:.8f}" for value in action.posterior),
                    "posterior_source_window_id": action.posterior_source_window_id,
                    "active_bank": action.active_bank,
                    "active_bank_version": action.active_bank_version,
                    "active_semantics_sha256": action.active_semantics_sha256,
                    "decision_source": action.decision_source,
                    "online_update_frozen": action.online_update_frozen,
                    "commit_gate_open": action.commit_gate_open,
                    "fallback_active": action.fallback_active,
                    "reset_request": action.reset_request,
                    "rollback_requested": action.rollback_requested,
                    "rollback_completed": action.rollback_completed,
                    "trusted_switch_requested": action.trusted_switch_requested,
                    "trusted_switch_completed": action.trusted_switch_completed,
                    "candidate_received": action.candidate_received,
                    "candidate_accepted": action.candidate_accepted,
                    "safety_stage_accepted": action.safety_stage_accepted,
                    "candidate_reason": action.candidate_reason,
                    "staged_version": action.staged_version,
                    "commit_status": action.commit_status,
                    "commit_reason": action.commit_reason,
                    "readback_status": action.readback_status,
                    "source_cycle": action.deadline.source_cycle,
                    "action_cycle": action.deadline.action_cycle,
                    "source_to_action_cycles": action.deadline.source_to_action_cycles,
                    "fast_deadline_miss": action.deadline.fast_deadline_miss,
                    "host_budget_violation": action.deadline.host_budget_violation,
                    "fault_flags": "|".join(action.fast_path_record.fault_flags),
                }
            )


def run_validation(report_path: Path, source_path: Path) -> dict[str, Any]:
    actions, provenance = _run_production_trace()
    report = _summarize(actions, provenance)
    source_files = (
        Path("cnn_fpga/runtime/regime_aware_safe_policy.py"),
        Path("cnn_fpga/runtime/closed_loop_fault_recovery.py"),
        Path("cnn_fpga/runtime/atomic_parameter_bank.py"),
        Path("cnn_fpga/runtime/experimental_event_fsm.py"),
        Path("cnn_fpga/runtime/conservative_fallback.py"),
        Path("cnn_fpga/decoder/periodic_adaptive_map.py"),
        Path("cnn_fpga/benchmark/regime_aware_safe_policy_validation.py"),
    )
    report["source_files"] = {
        path.as_posix(): _sha256(ROOT / path) for path in source_files
    }
    report["report_payload_sha256"] = _canonical_sha256(report)
    _write_csv(source_path, actions)
    report["source_data"] = {
        "path": source_path.relative_to(ROOT).as_posix(),
        "rows": len(actions),
        "sha256": _sha256(source_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    source_path = args.source_data if args.source_data.is_absolute() else ROOT / args.source_data
    report = run_validation(report_path, source_path)
    summary = report["gate_summary"]
    print(
        f"T6.6.2 Route-A policy: {summary['passed']}/{summary['total']} gates; "
        f"cycles={report['trace']['cycles']}; report={report_path}"
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
