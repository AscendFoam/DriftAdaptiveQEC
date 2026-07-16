"""T4.1.3 hybrid slow-state output and atomic-bank validation.

The validation restores the T4.1.1 selected Gaussian HMM, joins it with the
T4.1.2 experimental history and emits future-only hybrid outputs.  Nominal and
stress scheduler lanes exercise both stage-candidate and hold-active behavior.
No simulator truth field is written to output Source Data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cnn_fpga.benchmark.experimental_history_validation import (
    _communication_available,
    _crc_ok,
    _make_stream,
    _scheduler as _stress_scheduler,
)
from cnn_fpga.data.experimental_history import (
    DeployableLLRContext,
    ExperimentalHistoryBuilder,
    ExperimentalHistoryConfig,
    ObservedActionRecord,
    runtime_status_from_scheduler,
)
from cnn_fpga.decoder.hybrid_state_output import (
    CONTINUOUS_PARAMETER_NAMES,
    FORBIDDEN_DIRECT_OUTPUT_TOKENS,
    HybridStateEstimator,
    HybridStateEstimatorConfig,
    HybridStateOutput,
    ParameterBankRecommendation,
    schema_provenance,
    stage_parameter_bank_recommendation,
)
from cnn_fpga.decoder.regime_hmm import (
    GaussianRegimeHMM,
    REGIME_CLASSES,
    RegimeObservationWindow,
    summarize_regime_window,
)
from cnn_fpga.decoder.slow_loop_model_selection import RollingGaussianHMMAdapter
from cnn_fpga.runtime.latency_injector import LatencyInjector, StageLatencySpec
from cnn_fpga.runtime.param_bank import (
    DecoderRuntimeParams,
    ParamBank,
    ParameterUpdateConflictError,
)
from cnn_fpga.runtime.run_length_fsm import RunLengthFSMInput, RunLengthParameterBankFSM
from cnn_fpga.runtime.scheduler import DualLoopScheduler, SchedulerConfig


@dataclass(frozen=True)
class HybridStateValidationConfig:
    seeds: tuple[int, ...] = tuple(range(20261341, 20261349))
    cycles_per_seed: int = 2048
    history_cycles: int = 256
    output_stride_cycles: int = 32
    nominal_seed_count: int = 4
    bootstrap_replicates: int = 64
    block_length_cycles: int = 16
    checkpoint_path: str = "docs/t4_1_1_slow_loop_model_selection_checkpoints.pt"
    checkpoint_manifest_path: str = "docs/t4_1_1_slow_loop_model_selection_validation.json"

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        if len(seeds) < 6 or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must contain at least six unique values")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise TypeError("seeds must be nonnegative integers")
        for name, minimum in (
            ("cycles_per_seed", 512),
            ("history_cycles", 64),
            ("output_stride_cycles", 1),
            ("nominal_seed_count", 1),
            ("bootstrap_replicates", 32),
            ("block_length_cycles", 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        if self.history_cycles > self.cycles_per_seed:
            raise ValueError("history_cycles cannot exceed cycles_per_seed")
        if self.history_cycles % self.output_stride_cycles:
            raise ValueError("history_cycles must be divisible by output_stride_cycles")
        if not 0 < self.nominal_seed_count < len(seeds):
            raise ValueError("nominal_seed_count must leave both nominal and stress seeds")
        if len(seeds) * self.cycles_per_seed > 100_000:
            raise ValueError("validation workload must not exceed 100,000 cycles")


def _nominal_scheduler(seed: int) -> DualLoopScheduler:
    latency = LatencyInjector(
        dma=StageLatencySpec(8.0, 1.0, min_us=4.0),
        preprocess=StageLatencySpec(12.0, 1.5, min_us=6.0),
        inference=StageLatencySpec(35.0, 4.0, min_us=20.0),
        writeback=StageLatencySpec(8.0, 1.0, min_us=4.0),
        commit_ack=StageLatencySpec(3.0, 0.5, min_us=1.0),
        fast_cycle=StageLatencySpec(0.65, 0.08, min_us=0.2),
        seed=seed + 810_000,
    )
    return DualLoopScheduler(
        SchedulerConfig(
            t_fast_us=5.0,
            window_size=64,
            window_stride=32,
            slow_update_period_us=80.0,
            max_pending_windows=3,
            commit_delay_cycles=2,
            fast_path_budget_us=1.2,
            slow_path_budget_us=100.0,
            guard_cycles_after_commit=1,
            window_deadline_us=150.0,
        ),
        latency_injector=latency,
    )


def _hmm_from_mapping(mapping: dict[str, object]) -> GaussianRegimeHMM:
    required_arrays = (
        "standardization_mean",
        "standardization_scale",
        "emission_means",
        "emission_covariances",
        "emission_precisions",
        "emission_log_determinants",
        "transition_matrix",
        "initial_probabilities",
        "class_prior_probabilities",
    )
    missing = sorted(set(required_arrays + ("covariance_regularization", "transition_smoothing")) - set(mapping))
    if missing:
        raise ValueError(f"HMM checkpoint mapping is missing {missing}")
    return GaussianRegimeHMM(
        **{name: np.asarray(mapping[name], dtype=np.float64) for name in required_arrays},
        covariance_regularization=float(mapping["covariance_regularization"]),
        transition_smoothing=float(mapping["transition_smoothing"]),
    )


def _load_registered_hmm(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
) -> tuple[GaussianRegimeHMM, float, dict[str, object]]:
    # torch stays validation-only; the deployable output contract has no torch dependency.
    import torch

    checkpoint_target = Path(checkpoint_path)
    manifest_target = Path(manifest_path)
    checkpoint_hash = hashlib.sha256(checkpoint_target.read_bytes()).hexdigest()
    manifest = json.loads(manifest_target.read_text(encoding="utf-8"))
    if manifest["checkpoint"]["sha256"] != checkpoint_hash:
        raise ValueError("T4.1.1 checkpoint hash does not match validation manifest")
    payload = torch.load(checkpoint_target, map_location="cpu", weights_only=False)
    if payload.get("selected_family_from_validation") != "gaussian_hmm":
        raise ValueError("registered T4.1.1 winner is not gaussian_hmm")
    hmm_payload = payload["models"]["gaussian_hmm"]
    if hmm_payload.get("family") != "gaussian_hmm":
        raise ValueError("gaussian_hmm checkpoint family is malformed")
    model = _hmm_from_mapping(hmm_payload["model"])
    temperature = float(hmm_payload["temperature"])
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("HMM checkpoint temperature must be finite and positive")
    return model, temperature, {
        "path": checkpoint_target.as_posix(),
        "sha256": checkpoint_hash,
        "manifest_path": manifest_target.as_posix(),
        "selected_family": payload["selected_family_from_validation"],
        "temperature": temperature,
    }


def _raw_regime_window(sample, window_index: int, stride: int) -> RegimeObservationWindow:
    names = {name: index for index, name in enumerate(sample.feature_names)}
    rows = sample.values[-stride:]
    cycles = sample.cycle_indices[-stride:]
    if np.any(sample.mask[-stride:] != 1.0) or not np.array_equal(
        cycles, np.arange(cycles[0], cycles[-1] + 1)
    ):
        raise ValueError("regime window must be a fully valid contiguous observed suffix")
    values = np.column_stack(
        (
            rows[:, names["residual_q"]],
            rows[:, names["residual_p"]],
            rows[:, names["syndrome_x_e"]],
            rows[:, names["syndrome_z_e"]],
            np.maximum(rows[:, names["syndrome_x_leakage"]], rows[:, names["syndrome_z_leakage"]]),
            cycles & 1,
            rows[:, names["valid"]],
            np.minimum(rows[:, names["fast_deadline_ok"]], rows[:, names["slow_deadline_ok"]]),
        )
    )
    return RegimeObservationWindow(window_index, int(cycles[0]), values)


def _flatten_output(
    seed: int,
    lane: str,
    output: HybridStateOutput,
    *,
    staged_version: int | None,
) -> dict[str, object]:
    recommendation = output.parameter_bank_recommendation
    row: dict[str, object] = {
        "seed": seed,
        "lane": lane,
        "output_sequence": output.output_sequence,
        "as_of_cycle": output.as_of_cycle,
        "history_start_cycle": output.history_start_cycle,
        "regime_most_likely": output.regime.most_likely,
        "regime_entropy_nats": output.regime.entropy_nats,
        "leakage_probability_next_cycle": output.risk.leakage_probability_next_cycle,
        "leakage_probability_horizon": output.risk.leakage_probability_horizon,
        "expected_recovery_depth": output.risk.expected_recovery_depth,
        "ood_score": output.uncertainty.ood_score,
        "recommendation_confidence": output.uncertainty.recommendation_confidence,
        "bank_action": recommendation.bank_action,
        "recommended_mode": recommendation.recommended_mode,
        "base_active_version": recommendation.base_active_version,
        "valid_from_cycle": recommendation.valid_from_cycle,
        "expires_after_cycle": recommendation.expires_after_cycle,
        "recommendation_id": recommendation.recommendation_id,
        "payload_crc32": recommendation.payload_crc32,
        "staged_version": "" if staged_version is None else staged_version,
        "gain_qq": recommendation.gain_matrix[0][0],
        "gain_qp": recommendation.gain_matrix[0][1],
        "gain_pq": recommendation.gain_matrix[1][0],
        "gain_pp": recommendation.gain_matrix[1][1],
        "bias_q": recommendation.bias[0],
        "bias_p": recommendation.bias[1],
        "uncertainty_min_eigenvalue": float(
            np.min(np.linalg.eigvalsh(np.asarray(output.uncertainty.continuous_covariance)))
        ),
    }
    row.update(
        {name: float(getattr(output.continuous, name)) for name in CONTINUOUS_PARAMETER_NAMES}
    )
    row.update(
        {f"regime_p_{name}": output.regime.probabilities[index] for index, name in enumerate(REGIME_CLASSES)}
    )
    row.update(
        {
            f"recovery_depth_p_{index}": value
            for index, value in enumerate(output.risk.recovery_burden_posterior)
        }
    )
    row.update(
        {
            f"continuous_se_{name}": output.uncertainty.standard_errors[index]
            for index, name in enumerate(CONTINUOUS_PARAMETER_NAMES)
        }
    )
    return row


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        "cnn_fpga/decoder/hybrid_state_output.py",
        "cnn_fpga/benchmark/hybrid_state_output_validation.py",
        "cnn_fpga/data/experimental_history.py",
        "cnn_fpga/decoder/regime_hmm.py",
        "cnn_fpga/decoder/slow_loop_model_selection.py",
        "cnn_fpga/decoder/param_mapper.py",
        "cnn_fpga/runtime/param_bank.py",
    ):
        digest.update(path.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_hybrid_state_output_validation(
    config: HybridStateValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = HybridStateValidationConfig() if config is None else config
    if not isinstance(settings, HybridStateValidationConfig):
        raise TypeError("config must be HybridStateValidationConfig")
    model, temperature, checkpoint = _load_registered_hmm(
        settings.checkpoint_path, settings.checkpoint_manifest_path
    )
    rows: list[dict[str, object]] = []
    source_digest = hashlib.sha256()
    trace_hashes: dict[str, str] = {}
    action_counts = {"stage_candidate": 0, "hold_active": 0}
    lane_counts = {"nominal": 0, "stress": 0}
    mode_counts: dict[str, int] = {}
    commits = 0
    max_commit_parameter_error = 0.0
    first_stage_output: HybridStateOutput | None = None
    first_hold_output: HybridStateOutput | None = None
    replay_probe: tuple[object, np.ndarray, DecoderRuntimeParams, int, HybridStateOutput] | None = None

    for seed_index, seed in enumerate(settings.seeds):
        lane = "nominal" if seed_index < settings.nominal_seed_count else "stress"
        stream = _make_stream(seed, settings.cycles_per_seed)
        scheduler = _nominal_scheduler(seed) if lane == "nominal" else _stress_scheduler(seed)
        fsm = RunLengthParameterBankFSM()
        history_builder = ExperimentalHistoryBuilder(
            ExperimentalHistoryConfig(history_cycles=settings.history_cycles)
        )
        llr_context = DeployableLLRContext((0.34, 0.36), estimator_version=1)
        hmm = RollingGaussianHMMAdapter(
            model, history_windows=settings.history_cycles // settings.output_stride_cycles, temperature=temperature
        )
        estimator = HybridStateEstimator(
            HybridStateEstimatorConfig(
                minimum_samples=max(64, settings.history_cycles // 2),
                bootstrap_replicates=settings.bootstrap_replicates,
                block_length_cycles=settings.block_length_cycles,
                calibration_scope="registered_synthetic_pilot",
                ood_hold_threshold=0.58,
                minimum_regime_confidence=0.35,
                recommendation_ttl_cycles=settings.output_stride_cycles,
                bootstrap_seed=seed + 900_000,
            )
        )
        output_bank = ParamBank()
        trace_digest = hashlib.sha256()
        posterior = np.full(len(REGIME_CLASSES), 1.0 / len(REGIME_CLASSES))
        window_index = 0

        for cycle, step in enumerate(stream.steps):
            committed = output_bank.commit_if_ready(cycle)
            if committed is not None:
                commits += 1
            observed = step.observed
            events = []
            if cycle > 0:
                communication = True if lane == "nominal" else _communication_available(cycle, seed)
                events.extend(
                    scheduler.tick(
                        window_payload=observed.as_deployable_dict(),
                        communication_available=communication,
                    )
                )
                if lane == "stress" and cycle % 389 in (70, 71):
                    _, external = scheduler.stage_external_update(
                        DecoderRuntimeParams.identity(),
                        commit_epoch=cycle + 25,
                        metadata={"update_family": "registered_external_probe"},
                    )
                    events.extend(external)
            crc_ok = True if lane == "nominal" else _crc_ok(cycle, seed)
            runtime = runtime_status_from_scheduler(cycle, events, scheduler.snapshot(), crc_ok=crc_ok)
            decision = fsm.step(
                RunLengthFSMInput(
                    cycle_index=cycle,
                    residual=observed.residual_syndrome,
                    syndrome_x=observed.syndrome.x,
                    syndrome_z=observed.syndrome.z,
                    quadrature_phase_bit=cycle & 1,
                    valid=observed.valid,
                    crc_ok=crc_ok,
                    parameter_fresh=runtime.communication_available
                    and runtime.update_status not in {"failed", "stale"},
                    deadline_ok=runtime.fast_deadline_ok and runtime.slow_deadline_ok,
                )
            )
            sample = history_builder.append(
                observed,
                ObservedActionRecord.from_fsm_decision(decision),
                llr_context,
                runtime,
            )
            if (cycle + 1) % settings.output_stride_cycles == 0:
                summary = summarize_regime_window(
                    _raw_regime_window(sample, window_index, settings.output_stride_cycles)
                )
                posterior = hmm.step(summary)
                window_index += 1
            if cycle + 1 < settings.history_cycles or (cycle + 1) % settings.output_stride_cycles:
                continue

            active_before = output_bank.read_active()
            output = estimator.estimate(
                sample,
                posterior,
                regime_source="t4.1.1_registered_gaussian_hmm",
                active_params=active_before,
                active_bank_version=output_bank.active_version,
            )
            recommendation = output.parameter_bank_recommendation
            staged_version = None
            if recommendation.bank_action == "stage_candidate":
                pending = stage_parameter_bank_recommendation(output, output_bank, staged_cycle=cycle)
                staged_version = pending.version
                staged = output_bank.read_staging()
                max_commit_parameter_error = max(
                    max_commit_parameter_error,
                    float(np.max(np.abs(staged.K - np.asarray(recommendation.gain_matrix)))),
                    float(np.max(np.abs(staged.b - np.asarray(recommendation.bias)))),
                )
                if first_stage_output is None:
                    first_stage_output = output
            elif first_hold_output is None:
                first_hold_output = output
            action_counts[recommendation.bank_action] += 1
            lane_counts[lane] += 1
            mode_counts[recommendation.recommended_mode] = mode_counts.get(recommendation.recommended_mode, 0) + 1
            source_row = _flatten_output(seed, lane, output, staged_version=staged_version)
            rows.append(source_row)
            encoded = (json.dumps(source_row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            source_digest.update(encoded)
            trace_digest.update(encoded)
            if replay_probe is None:
                replay_probe = (sample, np.array(posterior, copy=True), active_before, recommendation.base_active_version, output)

        final_commit = output_bank.commit_if_ready(settings.cycles_per_seed)
        if final_commit is not None:
            commits += 1
        trace_hashes[str(seed)] = trace_digest.hexdigest()

    expected_outputs_per_seed = 1 + (
        settings.cycles_per_seed - settings.history_cycles
    ) // settings.output_stride_cycles
    if first_stage_output is None or first_hold_output is None or replay_probe is None:
        raise RuntimeError("validation did not exercise both stage and hold outputs")

    # Independent deterministic replay at output sequence zero.
    replay_sample, replay_posterior, replay_params, replay_version, original_output = replay_probe
    replay_estimator = HybridStateEstimator(
        HybridStateEstimatorConfig(
            minimum_samples=max(64, settings.history_cycles // 2),
            bootstrap_replicates=settings.bootstrap_replicates,
            block_length_cycles=settings.block_length_cycles,
            calibration_scope="registered_synthetic_pilot",
            ood_hold_threshold=0.58,
            minimum_regime_confidence=0.35,
            recommendation_ttl_cycles=settings.output_stride_cycles,
            bootstrap_seed=int(rows[0]["seed"]) + 900_000,
        )
    )
    replayed = replay_estimator.estimate(
        replay_sample,
        replay_posterior,
        regime_source="t4.1.1_registered_gaussian_hmm",
        active_params=replay_params,
        active_bank_version=replay_version,
    )
    deterministic_replay = replayed.to_deployable_dict() == original_output.to_deployable_dict()

    # Transactional negative probes use the first base-version-zero proposal.
    probe_bank = ParamBank()
    stage_parameter_bank_recommendation(first_stage_output, probe_bank, staged_cycle=first_stage_output.as_of_cycle)
    conflict_rejected = False
    try:
        stage_parameter_bank_recommendation(first_stage_output, probe_bank, staged_cycle=first_stage_output.as_of_cycle)
    except ParameterUpdateConflictError:
        conflict_rejected = True
    probe_bank.commit_if_ready(first_stage_output.parameter_bank_recommendation.valid_from_cycle)
    stale_rejected = False
    try:
        stage_parameter_bank_recommendation(first_stage_output, probe_bank, staged_cycle=first_stage_output.as_of_cycle)
    except ValueError:
        stale_rejected = True
    hold_rejected = False
    try:
        stage_parameter_bank_recommendation(first_hold_output, ParamBank(), staged_cycle=first_hold_output.as_of_cycle)
    except ValueError:
        hold_rejected = True
    crc_rejected = False
    try:
        recommendation = first_stage_output.parameter_bank_recommendation
        replace(recommendation, payload_crc32=recommendation.payload_crc32 ^ 1)
    except ValueError:
        crc_rejected = True

    source_fields = tuple(rows[0]) if rows else ()
    direct_tokens_in_source = sorted(
        {
            token
            for field in source_fields
            for token in FORBIDDEN_DIRECT_OUTPUT_TOKENS
            if token in "".join(character for character in field.lower() if character.isalnum())
        }
    )
    truth_tokens = ("truth", "hidden", "logical", "oracle", "teacher", "label")
    truth_fields = sorted(
        field
        for field in source_fields
        if any(token in "".join(character for character in field.lower() if character.isalnum()) for token in truth_tokens)
    )
    outputs_expected = len(settings.seeds) * expected_outputs_per_seed
    row_values_finite = all(
        all(
            np.isfinite(float(value))
            for key, value in row.items()
            if key not in {
                "lane",
                "regime_most_likely",
                "bank_action",
                "recommended_mode",
                "recommendation_id",
                "staged_version",
            }
        )
        for row in rows
    )
    posterior_normalized = all(
        abs(sum(float(row[f"regime_p_{name}"]) for name in REGIME_CLASSES) - 1.0) < 1.0e-10
        and abs(sum(float(row[f"recovery_depth_p_{index}"]) for index in range(7)) - 1.0) < 1.0e-10
        for row in rows
    )
    gates = {
        "registered_t4_1_1_hmm_checkpoint_hash_and_winner_match": checkpoint["selected_family"] == "gaussian_hmm",
        "output_grid_is_complete": len(rows) == outputs_expected,
        "seed_output_traces_are_unique": len(set(trace_hashes.values())) == len(settings.seeds),
        "hybrid_schema_has_every_required_family_and_no_direct_action": (
            not schema_provenance()["has_forbidden_direct_output"] and not direct_tokens_in_source
        ),
        "source_data_has_no_truth_hidden_logical_or_training_label": not truth_fields,
        "all_outputs_are_finite_and_normalized": row_values_finite and posterior_normalized,
        "all_uncertainty_covariances_are_psd": min(float(row["uncertainty_min_eigenvalue"]) for row in rows) >= -1.0e-10,
        "future_only_alignment_and_stride_are_exact": all(
            int(row["valid_from_cycle"]) == int(row["as_of_cycle"]) + 1
            and (int(row["as_of_cycle"]) + 1 - settings.history_cycles) % settings.output_stride_cycles == 0
            for row in rows
        ),
        "nominal_and_stress_lanes_both_produce_outputs": min(lane_counts.values()) > 0,
        "stage_and_hold_paths_are_both_exercised": min(action_counts.values()) > 0,
        "atomic_staging_preserves_parameters_and_every_stage_commits": (
            max_commit_parameter_error <= 1.0e-15 and commits == action_counts["stage_candidate"]
        ),
        "conflict_stale_hold_and_crc_negative_paths_reject": (
            conflict_rejected and stale_rejected and hold_rejected and crc_rejected
        ),
        "moving_block_output_replay_is_deterministic": deterministic_replay,
        "continuous_and_risk_outputs_have_nonzero_variation": (
            np.std([float(row["sigma_q"]) for row in rows]) > 0.0
            and np.std([float(row["leakage_probability_horizon"]) for row in rows]) > 0.0
            and np.std([float(row["expected_recovery_depth"]) for row in rows]) > 0.0
        ),
        "all_five_recommendation_profiles_are_exercised": set(mode_counts)
        == {"normal", "x_recovery", "z_recovery", "leakage_hold", "fallback"},
        "recommendation_ids_are_unique_within_output_grid": len({row["recommendation_id"] for row in rows}) == len(rows),
        "scope_is_registered_synthetic_pilot_not_hardware": schema_provenance()["hardware_measured"] is False,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t4.1.3-hybrid-state-output-validation-v1",
        "task_id": "T4.1.3",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "source_rows_sha256": source_digest.hexdigest(),
        "validation_config": asdict(settings),
        "registered_hmm_checkpoint": checkpoint,
        "output_schema": schema_provenance(),
        "aggregate": {
            "outputs": len(rows),
            "outputs_per_seed": expected_outputs_per_seed,
            "lane_counts": lane_counts,
            "bank_action_counts": action_counts,
            "recommended_mode_counts": mode_counts,
            "atomic_commits": commits,
            "max_commit_parameter_error": max_commit_parameter_error,
            "trace_sha256": trace_hashes,
            "ood_score_range": [min(float(row["ood_score"]) for row in rows), max(float(row["ood_score"]) for row in rows)],
            "leakage_horizon_range": [
                min(float(row["leakage_probability_horizon"]) for row in rows),
                max(float(row["leakage_probability_horizon"]) for row in rows),
            ],
            "expected_recovery_depth_range": [
                min(float(row["expected_recovery_depth"]) for row in rows),
                max(float(row["expected_recovery_depth"]) for row in rows),
            ],
        },
        "negative_probes": {
            "pending_bank_conflict_rejected": conflict_rejected,
            "stale_base_version_rejected": stale_rejected,
            "hold_active_stage_rejected": hold_rejected,
            "crc_corruption_rejected": crc_rejected,
        },
        "source_schema_audit": {
            "fields": list(source_fields),
            "direct_action_token_hits": direct_tokens_in_source,
            "truth_field_hits": truth_fields,
        },
        "gate_summary": {
            "passed": sum(gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": "future-only observed hybrid software state with block-bootstrap uncertainty and versioned CRC-bound inactive-bank recommendation",
            "forbidden": "per-cycle correction output, exact hidden recovery depth, calibrated physical risk, richer-input HMM retraining, logical/control gain, RTL or board measurement",
        },
    }
    return payload, rows


def write_hybrid_state_output_validation(
    json_path: str | Path = "docs/t4_1_3_hybrid_state_output_validation.json",
    csv_path: str | Path = "docs/t4_1_3_hybrid_state_output_source_data.csv",
    config: HybridStateValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_hybrid_state_output_validation(config)
    if not rows:
        raise RuntimeError("hybrid state validation produced no Source Data")
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="docs/t4_1_3_hybrid_state_output_validation.json")
    parser.add_argument("--csv", default="docs/t4_1_3_hybrid_state_output_source_data.csv")
    args = parser.parse_args(argv)
    payload = write_hybrid_state_output_validation(args.json, args.csv)
    print(json.dumps({"status": payload["status"], "gate_summary": payload["gate_summary"]}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HybridStateValidationConfig",
    "build_hybrid_state_output_validation",
    "write_hybrid_state_output_validation",
]
