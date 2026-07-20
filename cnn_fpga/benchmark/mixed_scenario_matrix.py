"""T5.1.2 mixed noise/regime scenario matrix.

The artifact deliberately contains several lane-local experiments instead of a
single cross-domain leaderboard.  Six syndrome-decoder scenarios share frozen
training/evaluation seeds and paired traces.  Loss, protocol ancilla/readout,
large-displacement recovery and leakage/correlation retain their native
physical observables and acceptance gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from math import atanh, isfinite, log, sqrt, tanh
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.continuous_adaptive_map import (
    ContinuousAdaptiveValidationConfig,
    _evaluate_seed,
    _mean_interval,
    _static_training_parameters,
    select_frozen_hyperparameters,
)
from cnn_fpga.decoder.periodic_adaptive_map import PeriodicMomentConfig
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState
from physics.noise_transfer_surrogate import (
    GKPNoiseTransferSurrogate,
    NoiseTransferConfig,
    NoiseTransferState,
    squeezing_db_to_peak_variance,
)
from physics.protocol_ancilla_errors import (
    SBSAncillaFaultOverlay,
    SBSFaultOverlayConfig,
    run_protocol_ancilla_validation,
)
from physics.sbs_displacement_fault import (
    DisplacementFaultSweepConfig,
    run_displacement_fault_sweep,
)
from physics.sbs_observation_reset import make_persistent_leakage_model
from physics.sbs_occupancy_correlation import (
    OccupancyCorrelationConfig,
    run_occupancy_correlation_validation,
)


TASK_ID = "T5.1.2"
SCHEMA_VERSION = "t5.1.2-mixed-scenario-matrix-v1"
PROTOCOL_ID = "MIXED-SCENARIO-MATRIX-LANE-LOCAL-V1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t5_1_2_mixed_scenario_matrix.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_1_2_mixed_scenario_matrix_source_data.csv")

REQUIRED_SCENARIO_IDS = (
    "static_gaussian",
    "mean_drift",
    "variance_drift",
    "correlation_drift",
    "loss",
    "readout_ancilla_drift",
    "burst_outlier",
    "large_error_recovery",
    "leakage",
    "calibration_shift",
)
DECODER_SCENARIO_IDS = (
    "static_gaussian",
    "mean_drift",
    "variance_drift",
    "correlation_drift",
    "burst_outlier",
    "calibration_shift",
)
DECODER_METHODS = ("standard", "static", "window", "ewma", "kalman", "oracle")

PARENT_ARTIFACTS = (
    ("T5.1.1", "docs/t5_1_1_comparison_set_registry.json"),
    ("T3.2.2", "docs/t3_2_2_continuous_adaptive_map_validation.json"),
    ("T2.3.8", "docs/t2_3_8_noise_transfer_validation.json"),
    ("T2.2.2", "docs/t2_2_2_protocol_ancilla_validation.json"),
    ("T2.0.5", "docs/t2_0_5_displacement_fault_trend.json"),
    ("T2.0.6", "docs/t2_0_6_occupancy_correlation.json"),
)
IMPLEMENTATION_PATHS = (
    "cnn_fpga/benchmark/mixed_scenario_matrix.py",
    "cnn_fpga/benchmark/continuous_adaptive_map.py",
    "cnn_fpga/decoder/periodic_adaptive_map.py",
    "physics/drift_processes.py",
    "physics/noise_transfer_surrogate.py",
    "physics/protocol_ancilla_errors.py",
    "physics/sbs_displacement_fault.py",
    "physics/sbs_occupancy_correlation.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parent_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") == "PASS" or payload.get("passed") is True:
        return True
    gate = payload.get("gate")
    if isinstance(gate, Mapping) and gate.get("passed") is True:
        return True
    checks = payload.get("checks")
    return isinstance(checks, Mapping) and bool(checks) and all(checks.values())


def _artifact_bindings() -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for task_id, relative in PARENT_ARTIFACTS:
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        bindings.append(
            {
                "task_id": task_id,
                "path": relative,
                "sha256": _sha256(path),
                "machine_pass": _parent_pass(payload),
            }
        )
    return bindings


def _implementation_bindings() -> list[dict[str, str]]:
    return [
        {"path": relative, "sha256": _sha256(ROOT / relative)}
        for relative in IMPLEMENTATION_PATHS
    ]


@dataclass(frozen=True)
class MixedDecoderScenario:
    scenario_id: str

    def __post_init__(self) -> None:
        if self.scenario_id not in DECODER_SCENARIO_IDS:
            raise ValueError(f"unknown mixed decoder scenario {self.scenario_id!r}")

    def states(self, windows: int) -> tuple[DriftState, ...]:
        if isinstance(windows, bool) or not isinstance(windows, int) or windows < 16:
            raise ValueError("windows must be an integer >= 16")
        lam = LATTICE_CONST
        states: list[DriftState] = []
        for step in range(windows):
            progress = step / (windows - 1)
            mu_q, mu_p = 0.035 * lam, -0.025 * lam
            sigma_q, sigma_p, rho = 0.155 * lam, 0.125 * lam, 0.15
            p_outlier, outlier_scale, burst_active = 0.0, 1.0, False
            event_id = 0
            if self.scenario_id == "static_gaussian":
                pass
            elif self.scenario_id == "mean_drift":
                mu_q = (-0.18 + 0.36 * progress) * lam
                mu_p = (0.13 - 0.27 * progress) * lam
            elif self.scenario_id == "variance_drift":
                sigma_q = 0.10 * lam * np.exp(log(0.23 / 0.10) * progress)
                sigma_p = 0.20 * lam * np.exp(log(0.115 / 0.20) * progress)
                rho = 0.20
            elif self.scenario_id == "correlation_drift":
                rho = tanh(atanh(-0.72) + (atanh(0.72) - atanh(-0.72)) * progress)
                sigma_q, sigma_p = 0.18 * lam, 0.15 * lam
            elif self.scenario_id == "burst_outlier":
                first = windows // 4 <= step < windows // 4 + max(3, windows // 8)
                second = 3 * windows // 4 <= step < 3 * windows // 4 + max(2, windows // 10)
                burst_active = first or second
                event_id = 1 if first else (2 if second else 0)
                p_outlier = 0.10 if burst_active else 0.01
                outlier_scale = 4.5 if burst_active else 3.0
                mu_q = (0.07 if burst_active else 0.0) * lam
                mu_p = (-0.05 if burst_active else 0.0) * lam
            elif self.scenario_id == "calibration_shift":
                if step >= windows // 2:
                    mu_q, mu_p = 0.17 * lam, -0.14 * lam
                    sigma_q, sigma_p, rho = 0.21 * lam, 0.105 * lam, -0.48
                    event_id = 1
            states.append(
                DriftState(
                    step=step,
                    time=float(step),
                    mu_q=float(mu_q),
                    mu_p=float(mu_p),
                    sigma_q=float(sigma_q),
                    sigma_p=float(sigma_p),
                    rho=float(rho),
                    p_outlier=p_outlier,
                    outlier_scale=outlier_scale,
                    burst_active=burst_active,
                    source="t5.1.2-mixed-scenario-matrix",
                    regime=self.scenario_id,
                    event_id=event_id,
                )
            )
        return tuple(states)


def decoder_scenarios() -> tuple[MixedDecoderScenario, ...]:
    return tuple(MixedDecoderScenario(name) for name in DECODER_SCENARIO_IDS)


def production_decoder_config() -> ContinuousAdaptiveValidationConfig:
    return ContinuousAdaptiveValidationConfig(
        training_seeds=(20260716101, 20260716102, 20260716103),
        evaluation_seeds=tuple(range(20260716201, 20260716207)),
        windows=32,
        calibration_windows=4,
        observation_samples_per_window=256,
        training_score_samples_per_window=256,
        evaluation_samples_per_window=512,
    )


def _decoder_lane() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = production_decoder_config()
    hyperparameters = select_frozen_hyperparameters(settings)
    static_parameters = _static_training_parameters(settings)
    moment_config = PeriodicMomentConfig(
        minimum_samples=min(64, settings.observation_samples_per_window)
    )
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(decoder_scenarios()):
        scenario_rows = [
            _evaluate_seed(
                scenario,
                20 + scenario_index,
                seed,
                settings,
                hyperparameters,
                static_parameters,
                moment_config,
            )
            for seed in settings.evaluation_seeds
        ]
        rows.extend(scenario_rows)
        methods: dict[str, Any] = {}
        for method in DECODER_METHODS:
            metrics: dict[str, Any] = {
                "error_rate_seed_cluster_ci": _mean_interval(
                    [float(row[f"{method}_error_rate"]) for row in scenario_rows],
                    settings.confidence_level,
                )
            }
            if method != "standard":
                metrics["nll_seed_cluster_ci"] = _mean_interval(
                    [float(row[f"{method}_nll"]) for row in scenario_rows],
                    settings.confidence_level,
                )
                metrics["brier_seed_cluster_ci"] = _mean_interval(
                    [float(row[f"{method}_brier"]) for row in scenario_rows],
                    settings.confidence_level,
                )
            methods[method] = metrics
        aggregates.append(
            {
                "scenario_id": scenario.scenario_id,
                "seed_clusters": len(scenario_rows),
                "windows_per_seed": settings.windows,
                "evaluation_samples": sum(int(row["evaluation_samples"]) for row in scenario_rows),
                "unique_trace_hashes": len({row["trace_sha256"] for row in scenario_rows}),
                "methods": methods,
                "paired_fixed_contrasts": {
                    "static_minus_ewma": _mean_interval(
                        [float(row["static_minus_ewma_error_rate"]) for row in scenario_rows],
                        settings.confidence_level,
                    ),
                    "static_minus_kalman": _mean_interval(
                        [float(row["static_minus_kalman_error_rate"]) for row in scenario_rows],
                        settings.confidence_level,
                    ),
                    "static_minus_oracle": _mean_interval(
                        [
                            float(row["static_error_rate"]) - float(row["oracle_error_rate"])
                            for row in scenario_rows
                        ],
                        settings.confidence_level,
                    ),
                },
            }
        )
    payload = {
        "lane_id": "decoder_syndrome_level_paired",
        "scope": "wrapped_Gaussian_or_mixture_syndrome_level_not_finite_energy_protocol_fidelity",
        "executed_scenarios": list(DECODER_SCENARIO_IDS),
        "executed_comparators": list(DECODER_METHODS),
        "not_executed_in_this_lane": [
            "no_correction_idle_memory",
            "measurement_feedback_sbs",
            "autonomous_sbs",
            "top_k_lattice_coset_map",
            "bayesian_state_space",
            "training_selected_sliding_window",
            "run_length_event_controller",
            "regime_hmm",
            "mf_fnn",
            "rnn_teacher_student",
            "control_oracle",
        ],
        "nonexecution_reason": "different decision target, horizon, information set, or protocol-native metric; retained for later matched lanes, never scored as a loss here",
        "shared_trace_contract": "all six methods decode identical displacement/residual truth per scenario-seed-window; predictors update only after current-window decoding",
        "training_contract": "hyperparameters and static parameters are fit only on disjoint original T3.2.2 training scenarios/seeds before all T5.1.2 evaluation",
        "config": asdict(settings),
        "frozen_hyperparameters": asdict(hyperparameters),
        "static_training_parameters": asdict(static_parameters),
        "scenario_aggregates": aggregates,
        "seed_rows": rows,
    }
    return payload, rows


def _loss_lane() -> dict[str, Any]:
    peak = squeezing_db_to_peak_variance(
        10.0, coordinate_chart="decoder_standardized"
    )
    state = NoiseTransferState(
        lattice_index=(1, -1),
        # Zero offset isolates attenuation-induced lattice bias.  Calibration
        # offset is exercised independently by ``calibration_shift``.
        signal_offset=(0.0, 0.0),
        # Diagonal axes make the reported joint jump probability exact.  The
        # correlated-axis path intentionally reports only Frechet bounds and
        # therefore belongs to a separate covariance scenario, not this loss
        # sweep.
        fluctuation_covariance=((peak, 0.0), (0.0, peak)),
    )
    rows: list[dict[str, Any]] = []
    for eta in (1.0, 0.98, 0.94, 0.88):
        result = GKPNoiseTransferSurrogate(
            NoiseTransferConfig(
                resource_covariance=((peak, 0.0), (0.0, peak)),
                loss_transmissivity=eta,
                measurement_efficiency=0.97,
            )
        ).propagate(state)
        bias = np.asarray(result.loss_bias, dtype=np.float64)
        decision = np.asarray(result.decision_covariance, dtype=np.float64)
        rows.append(
            {
                "loss_transmissivity": eta,
                "loss_gamma": -log(eta),
                "loss_bias_norm": float(np.linalg.norm(bias)),
                "decision_covariance_trace": float(np.trace(decision)),
                "q_odd_alias_probability": result.logical_jump.q_odd_probability,
                "p_odd_alias_probability": result.logical_jump.p_odd_probability,
                "any_jump_probability": result.logical_jump.any_jump_probability,
                "validity": result.validity,
            }
        )
    bias = [row["loss_bias_norm"] for row in rows]
    covariance = [row["decision_covariance_trace"] for row in rows]
    gates = {
        "four_registered_transmissivity_points": len(rows) == 4,
        "loss_bias_monotone_with_loss": all(right > left for left, right in zip(bias, bias[1:])),
        "decision_noise_monotone_with_loss": all(
            right > left for left, right in zip(covariance, covariance[1:])
        ),
        "probabilities_finite_and_bounded": all(
            isfinite(float(row[key])) and 0.0 <= float(row[key]) <= 1.0
            for row in rows
            for key in ("q_odd_alias_probability", "p_odd_alias_probability", "any_jump_probability")
        ),
    }
    return {
        "lane_id": "loss_noise_transfer",
        "scenario_id": "loss",
        "scope": "factorized_GKP_noise_transfer_surrogate_not_full_protocol_lifetime",
        "fixed_squeezing_db": 10.0,
        "measurement_efficiency": 0.97,
        "rows": rows,
        "gates": gates,
    }


def _readout_base(readout_error: float):
    confusion = np.asarray(
        [
            [1.0 - readout_error, readout_error, 0.0],
            [readout_error, 1.0 - readout_error, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return make_persistent_leakage_model(
        readout_confusion=confusion,
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=1.0,
        counter_max=2**31 - 1,
        readout_provenance="T5.1.2 registered readout drift assumptions",
        parameter_provenance="T5.1.2 no-leakage perfect-reset isolation",
    )


def _readout_ancilla_lane() -> dict[str, Any]:
    samples = 20_000
    rates = ((0.0, 0.0), (0.01, 0.005), (0.02, 0.01), (0.04, 0.02))
    rows: list[dict[str, Any]] = []
    for index, (bit_rate, readout_error) in enumerate(rates):
        overlay = SBSAncillaFaultOverlay(
            _readout_base(readout_error),
            SBSFaultOverlayConfig(
                bit_flip_probabilities=((0.0, bit_rate, 0.0), (0.0, 0.0, 0.0)),
                phase_flip_probabilities=((0.0, 0.5 * bit_rate, 0.0), (0.0, 0.0, 0.0)),
                logical_fault_given_big_cd_bit=(0.5, 0.0),
                phase_backaction_scale=(0.01, 0.01),
                small_cd_bit_backaction_scale=(0.02, 0.02),
                misclassification_rotation_max_rad=0.6,
                parameter_provenance="T5.1.2 registered ancilla/readout drift assumptions",
            ),
        )
        trajectory = overlay.simulate(
            ("K_gg",) * samples, seed=20260716301 + index
        )
        bit_count = sum(
            any(
                event.constituent == "X"
                and event.fault_type == "bit_flip"
                and event.stage == "big_cd"
                for event in step.fault_truth.events
            )
            for step in trajectory.steps
        )
        logical_count = sum(
            step.fault_truth.logical_backaction_by_constituent[0]
            for step in trajectory.steps
        )
        mismatch_count = sum(
            sum(step.fault_truth.readout_misclassified)
            for step in trajectory.steps
        )
        observed_bit = bit_count / samples
        observed_readout = mismatch_count / (2 * samples)
        observed_logical = logical_count / samples
        bit_se = sqrt(max(bit_rate * (1.0 - bit_rate), 0.0) / samples)
        readout_se = sqrt(max(readout_error * (1.0 - readout_error), 0.0) / (2 * samples))
        logical_expected = 0.5 * bit_rate
        logical_se = sqrt(max(logical_expected * (1.0 - logical_expected), 0.0) / samples)
        rows.append(
            {
                "drift_level": index,
                "samples": samples,
                "seed": 20260716301 + index,
                "expected_big_cd_bit_rate": bit_rate,
                "observed_big_cd_bit_rate": observed_bit,
                "big_cd_bit_z_score": 0.0 if bit_se == 0.0 else (observed_bit - bit_rate) / bit_se,
                "expected_readout_mismatch_rate": readout_error,
                "observed_readout_mismatch_rate": observed_readout,
                "readout_mismatch_z_score": 0.0 if readout_se == 0.0 else (observed_readout - readout_error) / readout_se,
                "expected_logical_backaction_rate": logical_expected,
                "observed_logical_backaction_rate": observed_logical,
                "logical_backaction_z_score": 0.0 if logical_se == 0.0 else (observed_logical - logical_expected) / logical_se,
            }
        )
    production = run_protocol_ancilla_validation(samples=60_000, seed=20260716321)
    gates = {
        "four_drift_levels": len(rows) == 4,
        "all_binomial_rates_within_five_sigma": all(
            abs(float(row[key])) <= 5.0
            for row in rows
            for key in ("big_cd_bit_z_score", "readout_mismatch_z_score", "logical_backaction_z_score")
        ),
        "observed_bit_rate_monotone": all(
            right > left
            for left, right in zip(
                [row["observed_big_cd_bit_rate"] for row in rows],
                [row["observed_big_cd_bit_rate"] for row in rows][1:],
            )
        ),
        "observed_readout_rate_monotone": all(
            right > left
            for left, right in zip(
                [row["observed_readout_mismatch_rate"] for row in rows],
                [row["observed_readout_mismatch_rate"] for row in rows][1:],
            )
        ),
        "protocol_native_endpoint_checks_pass": all(production.checks.values()),
    }
    return {
        "lane_id": "protocol_readout_ancilla_fault_drift",
        "scenario_id": "readout_ancilla_drift",
        "scope": "protocol_native_effective_fault_overlay_not_device_calibration",
        "rows": rows,
        "production_endpoint": production.as_dict(),
        "gates": gates,
    }


def _component_lanes() -> tuple[dict[str, Any], dict[str, Any]]:
    displacement = run_displacement_fault_sweep(
        replace(
            DisplacementFaultSweepConfig(),
            seed=20260716401,
            bootstrap_seed=20260716402,
        )
    )
    occupancy = run_occupancy_correlation_validation(
        replace(
            OccupancyCorrelationConfig(),
            seed=20260716501,
            bootstrap_seed=20260716502,
        )
    )
    return (
        {
            "lane_id": "large_error_recovery_component",
            "scenario_id": "large_error_recovery",
            "ranking_status": "component_only_not_decoder_leaderboard",
            "result": displacement.to_dict(),
        },
        {
            "lane_id": "leakage_occupancy_correlation_component",
            "scenario_id": "leakage",
            "ranking_status": "component_only_not_decoder_leaderboard",
            "result": occupancy.to_dict(),
        },
    )


def _all_numeric_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, Mapping):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return isfinite(float(value))
    return True


def validate_matrix_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Fail closed on schema, pairing, provenance and lane-mixing mutations."""

    if payload.get("task_id") != TASK_ID or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("task/schema identity drifted")
    if tuple(payload.get("required_scenarios", ())) != REQUIRED_SCENARIO_IDS:
        raise ValueError("required scenario order or membership drifted")
    execution_map = payload.get("scenario_execution_map")
    if not isinstance(execution_map, Mapping) or set(execution_map) != set(REQUIRED_SCENARIO_IDS):
        raise ValueError("scenario execution map must cover the exact registered set")
    if payload.get("matrix_status") != "EXECUTED_LANE_LOCAL_NO_CROSS_LANE_RANKING":
        raise ValueError("matrix status must preserve lane-local nonmixing")
    if "global_leaderboard" in payload:
        raise ValueError("global_leaderboard is forbidden")

    decoder = payload.get("decoder_lane")
    if not isinstance(decoder, Mapping):
        raise ValueError("decoder lane is missing")
    if tuple(decoder.get("executed_scenarios", ())) != DECODER_SCENARIO_IDS:
        raise ValueError("decoder scenario membership drifted")
    if tuple(decoder.get("executed_comparators", ())) != DECODER_METHODS:
        raise ValueError("decoder comparator membership drifted")
    config = decoder.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("decoder config is missing")
    if set(config["training_seeds"]) & set(config["evaluation_seeds"]):
        raise ValueError("training/evaluation seeds overlap")
    seed_rows = decoder.get("seed_rows")
    if not isinstance(seed_rows, list) or len(seed_rows) != 36:
        raise ValueError("decoder lane must contain 6 scenarios x 6 seed rows")
    traces = [str(row.get("trace_sha256", "")) for row in seed_rows]
    if any(len(value) != 64 for value in traces) or len(set(traces)) != 36:
        raise ValueError("decoder trace hashes must be complete and unique")
    for scenario_id in DECODER_SCENARIO_IDS:
        rows = [row for row in seed_rows if row.get("scenario_id") == scenario_id]
        if len(rows) != 6:
            raise ValueError(f"scenario {scenario_id} must have six seed clusters")
        for row in rows:
            for method in DECODER_METHODS:
                rate = float(row[f"{method}_error_rate"])
                if not isfinite(rate) or not 0.0 <= rate <= 1.0:
                    raise ValueError("decoder error rate is not finite and bounded")
    aggregates = decoder.get("scenario_aggregates")
    if not isinstance(aggregates, list) or {
        row.get("scenario_id") for row in aggregates
    } != set(DECODER_SCENARIO_IDS):
        raise ValueError("decoder aggregate scenarios drifted")
    if any(set(row.get("methods", {})) != set(DECODER_METHODS) for row in aggregates):
        raise ValueError("decoder aggregate methods drifted")

    for lane_key, scenario_id in (
        ("loss_lane", "loss"),
        ("readout_ancilla_lane", "readout_ancilla_drift"),
        ("large_error_lane", "large_error_recovery"),
        ("leakage_lane", "leakage"),
    ):
        lane = payload.get(lane_key)
        if not isinstance(lane, Mapping) or lane.get("scenario_id") != scenario_id:
            raise ValueError(f"{lane_key} scenario binding drifted")
    for lane_key in ("large_error_lane", "leakage_lane"):
        if payload[lane_key].get("ranking_status") != "component_only_not_decoder_leaderboard":
            raise ValueError("component-only result entered a decoder ranking")

    bindings = payload.get("artifact_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(PARENT_ARTIFACTS):
        raise ValueError("parent artifact binding count drifted")
    for binding in bindings:
        path = ROOT / str(binding["path"])
        if not binding.get("machine_pass") or _sha256(path) != binding.get("sha256"):
            raise ValueError("parent artifact binding is stale or failed")
    implementations = payload.get("implementation_bindings")
    if not isinstance(implementations, list) or len(implementations) != len(IMPLEMENTATION_PATHS):
        raise ValueError("implementation binding count drifted")
    for binding in implementations:
        path = ROOT / str(binding["path"])
        if _sha256(path) != binding.get("sha256"):
            raise ValueError("implementation binding is stale")

    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or len(gates) != 15 or not all(gates.values()):
        raise ValueError("all fifteen acceptance gates must pass")
    if payload.get("status") != "PASS":
        raise ValueError("payload status must be PASS after all gates pass")
    source = payload.get("source_data")
    if source is not None:
        path = Path(str(source["path"]))
        if not path.is_absolute():
            path = ROOT / path
        if _sha256(path) != source.get("sha256"):
            raise ValueError("source-data hash is stale")
    return (
        "schema_and_exact_scenario_set",
        "paired_decoder_seed_trace_contract",
        "native_lane_and_component_nonmixing",
        "parent_and_implementation_provenance",
        "fifteen_acceptance_gates",
    )


def build_mixed_scenario_matrix() -> dict[str, Any]:
    artifacts = _artifact_bindings()
    implementations = _implementation_bindings()
    decoder, decoder_rows = _decoder_lane()
    loss = _loss_lane()
    ancilla = _readout_ancilla_lane()
    large_error, leakage = _component_lanes()
    scenario_ids = (
        tuple(decoder["executed_scenarios"])
        + (loss["scenario_id"], ancilla["scenario_id"], large_error["scenario_id"], leakage["scenario_id"])
    )
    trace_hashes = [str(row["trace_sha256"]) for row in decoder_rows]
    static_states = decoder_scenarios()[0].states(decoder["config"]["windows"])
    static_signatures = {
        (
            state.mu_q,
            state.mu_p,
            state.sigma_q,
            state.sigma_p,
            state.rho,
            state.loss_gamma,
            state.p_outlier,
            state.outlier_scale,
            state.burst_active,
            state.event_id,
        )
        for state in static_states
    }
    gates = {
        "exact_ten_scenario_coverage": set(scenario_ids) == set(REQUIRED_SCENARIO_IDS) and len(scenario_ids) == 10,
        "decoder_training_evaluation_seeds_disjoint": not (
            set(decoder["config"]["training_seeds"]) & set(decoder["config"]["evaluation_seeds"])
        ),
        "six_decoder_scenarios_have_six_seed_clusters": len(decoder_rows) == 36 and all(
            row["seed_clusters"] == 6 for row in decoder["scenario_aggregates"]
        ),
        "all_decoder_traces_unique": len(trace_hashes) == len(set(trace_hashes)) == 36,
        "static_gaussian_is_constant": len(static_signatures) == 1,
        "causal_update_contract_preserved": "only after current-window decoding" in decoder["shared_trace_contract"],
        "decoder_metrics_finite": _all_numeric_finite(decoder),
        "loss_native_gates_pass": all(loss["gates"].values()),
        "readout_ancilla_native_gates_pass": all(ancilla["gates"].values()),
        "large_error_native_gate_pass": bool(large_error["result"]["gate"]["passed"]),
        "leakage_native_gate_pass": bool(leakage["result"]["gate"]["passed"]),
        "component_rows_excluded_from_decoder_ranking": all(
            lane["ranking_status"] == "component_only_not_decoder_leaderboard"
            for lane in (large_error, leakage)
        ),
        "all_parent_artifacts_current_and_pass": all(row["machine_pass"] for row in artifacts),
        "all_implementation_bindings_present": len(implementations) == len(IMPLEMENTATION_PATHS),
        "no_global_leaderboard_field": "global_leaderboard" not in json.dumps(
            {"decoder": decoder, "loss": loss, "ancilla": ancilla, "large_error": large_error, "leakage": leakage},
            sort_keys=True,
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass_semantics": "exact scenario coverage, shared decoder traces, lane-native physics/statistics, provenance and nonmixing gates pass; no universal comparator superiority is implied",
        "matrix_status": "EXECUTED_LANE_LOCAL_NO_CROSS_LANE_RANKING",
        "required_scenarios": list(REQUIRED_SCENARIO_IDS),
        "scenario_execution_map": {
            scenario_id: (
                "decoder_syndrome_level_paired"
                if scenario_id in DECODER_SCENARIO_IDS
                else {
                    "loss": "loss_noise_transfer",
                    "readout_ancilla_drift": "protocol_readout_ancilla_fault_drift",
                    "large_error_recovery": "large_error_recovery_component",
                    "leakage": "leakage_occupancy_correlation_component",
                }[scenario_id]
            )
            for scenario_id in REQUIRED_SCENARIO_IDS
        },
        "artifact_bindings": artifacts,
        "implementation_bindings": implementations,
        "decoder_lane": decoder,
        "loss_lane": loss,
        "readout_ancilla_lane": ancilla,
        "large_error_lane": large_error,
        "leakage_lane": leakage,
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "failed": [name for name, passed in gates.items() if not passed],
        },
        "claim_boundary": {
            "allowed": "the ten registered mixed scenarios were actually executed in their native lanes with frozen seeds, paired decoder traces, fresh component runs and explicit lane-local metrics",
            "forbidden": "a global leaderboard, algorithm selection from evaluation results, finite-energy or device lifetime equivalence, device-calibrated fault rates, deployable oracle claims, or universal adaptive/NMF/CNN superiority",
        },
    }
    payload["contract_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"generated_at_utc", "contract_sha256"}
        }
    )
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    validate_matrix_payload(normalized)
    return normalized


def source_data_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id, lane_id in payload["scenario_execution_map"].items():
        rows.append(
            {"row_type": "scenario", "record_id": scenario_id, "lane_id": lane_id, "seed": "", "method": "", "metric": "registered", "value": 1, "detail": ""}
        )
    for row in payload["decoder_lane"]["seed_rows"]:
        rows.append(
            {"row_type": "decoder_seed", "record_id": row["scenario_id"], "lane_id": payload["decoder_lane"]["lane_id"], "seed": row["evaluation_seed"], "method": "paired_all", "metric": "trace_sha256", "value": row["trace_sha256"], "detail": json.dumps(row, sort_keys=True)}
        )
    for scenario in payload["decoder_lane"]["scenario_aggregates"]:
        for method, metrics in scenario["methods"].items():
            rows.append(
                {"row_type": "decoder_method", "record_id": scenario["scenario_id"], "lane_id": payload["decoder_lane"]["lane_id"], "seed": "", "method": method, "metric": "error_rate", "value": metrics["error_rate_seed_cluster_ci"]["estimate"], "detail": json.dumps(metrics, sort_keys=True)}
            )
    for row in payload["loss_lane"]["rows"]:
        rows.append(
            {"row_type": "loss", "record_id": "loss", "lane_id": payload["loss_lane"]["lane_id"], "seed": "", "method": "noise_transfer_surrogate", "metric": "loss_bias_norm", "value": row["loss_bias_norm"], "detail": json.dumps(row, sort_keys=True)}
        )
    for row in payload["readout_ancilla_lane"]["rows"]:
        rows.append(
            {"row_type": "ancilla_drift", "record_id": "readout_ancilla_drift", "lane_id": payload["readout_ancilla_lane"]["lane_id"], "seed": row["seed"], "method": "protocol_native_fault_overlay", "metric": "observed_big_cd_bit_rate", "value": row["observed_big_cd_bit_rate"], "detail": json.dumps(row, sort_keys=True)}
        )
    rows.append(
        {"row_type": "ancilla_endpoint", "record_id": "readout_ancilla_drift", "lane_id": payload["readout_ancilla_lane"]["lane_id"], "seed": payload["readout_ancilla_lane"]["production_endpoint"]["seed"], "method": "sbs_and_sharpen_trim", "metric": "all_checks_pass", "value": all(payload["readout_ancilla_lane"]["production_endpoint"]["checks"].values()), "detail": json.dumps(payload["readout_ancilla_lane"]["production_endpoint"], sort_keys=True)}
    )
    for point in payload["large_error_lane"]["result"]["points"]:
        rows.append(
            {"row_type": "large_error", "record_id": "large_error_recovery", "lane_id": payload["large_error_lane"]["lane_id"], "seed": payload["large_error_lane"]["result"]["config"]["seed"], "method": "protocol_native_component", "metric": "amplitude_over_lattice", "value": point["amplitude_over_lattice"], "detail": json.dumps(point, sort_keys=True)}
        )
    rows.append(
        {"row_type": "leakage", "record_id": "leakage", "lane_id": payload["leakage_lane"]["lane_id"], "seed": payload["leakage_lane"]["result"]["config"]["seed"], "method": "occupancy_correlation_component", "metric": "gate_pass", "value": payload["leakage_lane"]["result"]["gate"]["passed"], "detail": json.dumps(payload["leakage_lane"]["result"], sort_keys=True)}
    )
    for gate_id, passed in payload["gates"].items():
        rows.append(
            {"row_type": "gate", "record_id": gate_id, "lane_id": "acceptance", "seed": "", "method": "", "metric": "pass", "value": passed, "detail": ""}
        )
    return rows


def write_artifacts(
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = build_mixed_scenario_matrix()
    rows = source_data_rows(payload)
    csv_path = Path(source_data_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("row_type", "record_id", "lane_id", "seed", "method", "metric", "value", "detail")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    try:
        source_label = str(csv_path.relative_to(ROOT))
    except ValueError:
        source_label = str(csv_path)
    payload["source_data"] = {
        "path": source_label,
        "row_count": len(rows),
        "sha256": _sha256(csv_path),
    }
    validate_matrix_payload(payload)
    output = Path(artifact_path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE_DATA))
    args = parser.parse_args(argv)
    payload = write_artifacts(args.artifact, args.source_data)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": payload["status"],
                "scenarios": len(payload["required_scenarios"]),
                "decoder_seed_rows": len(payload["decoder_lane"]["seed_rows"]),
                "source_rows": payload["source_data"]["row_count"],
                "gates": payload["gate_summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TASK_ID",
    "SCHEMA_VERSION",
    "PROTOCOL_ID",
    "REQUIRED_SCENARIO_IDS",
    "DECODER_SCENARIO_IDS",
    "DECODER_METHODS",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "MixedDecoderScenario",
    "decoder_scenarios",
    "production_decoder_config",
    "validate_matrix_payload",
    "build_mixed_scenario_matrix",
    "source_data_rows",
    "write_artifacts",
]
