"""T4.3.1 executable three-timescale cadence and adaptation-lag validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.fast_path_fixed_point import BitAccurateFastPath, FastPathCodeInput
from cnn_fpga.runtime.latency_injector import LatencyInjector, StageLatencySpec
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig
from cnn_fpga.runtime.scheduler import DualLoopScheduler, SchedulerConfig
from cnn_fpga.runtime.three_timescale_cadence import (
    MODEL_SCOPE,
    ThreeTimescaleCadence,
    ThreeTimescaleCadenceConfig,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "cnn_fpga" / "config" / "hardware_hil.yaml"
DEFAULT_JSON = ROOT / "docs" / "t4_3_1_three_timescale_cadence_validation.json"
DEFAULT_SWEEP_CSV = ROOT / "docs" / "t4_3_1_adaptation_lag_phase_sweep.csv"
DEFAULT_TRACE_CSV = ROOT / "docs" / "t4_3_1_cadence_execution_trace.csv"
SCHEMA_VERSION = "t4.3.1-three-timescale-cadence-validation-v1"
TRACE_ONSET_EPOCH = 2040


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/three_timescale_cadence.py",
        "cnn_fpga/runtime/scheduler.py",
        "cnn_fpga/runtime/fast_path_fixed_point.py",
        "cnn_fpga/benchmark/three_timescale_cadence_validation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _row_hash(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty Source Data table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _production_contract() -> tuple[dict[str, Any], SchedulerConfig, ThreeTimescaleCadence]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    scheduler = SchedulerConfig.from_config(raw)
    latency = raw["latency_model"]
    service_us = sum(
        float(latency[f"{name}_mean_us"])
        for name in ("dma", "preprocess", "inference", "writeback", "commit_ack")
    )
    cadence = ThreeTimescaleCadence(
        ThreeTimescaleCadenceConfig(
            t_fast_us=scheduler.t_fast_us,
            window_size=scheduler.window_size,
            window_stride=scheduler.resolved_window_stride,
            slow_update_period_us=scheduler.slow_update_period_us,
            slow_service_us=service_us,
            commit_delay_cycles=scheduler.commit_delay_cycles,
            event_register_cycles=1,
            max_parameter_age_cycles=int(raw["runtime"]["max_parameter_age_cycles"]),
            recalibration_period_us=60_000_000.0,
        )
    )
    return raw, scheduler, cadence


def _constant_injector(*, service_us: float) -> LatencyInjector:
    zero = StageLatencySpec(mean_us=0.0, std_us=0.0, distribution="constant")
    service = StageLatencySpec(mean_us=service_us, std_us=0.0, distribution="constant")
    return LatencyInjector(
        dma=zero,
        preprocess=zero,
        inference=service,
        writeback=zero,
        commit_ack=zero,
        fast_cycle=zero,
        seed=43101,
    )


def _parameter_images() -> tuple[tuple[Any, ...], DecoderRuntimeParams, DecoderRuntimeParams]:
    config = ParametricMAPLUTConfig()
    profiles = registered_parameter_profiles(config)
    initial = profiles[0][0]
    proposed = profiles[1][0]
    images = (
        compile_parametric_map_lut(initial, active_bank_version=0, config=config),
        compile_parametric_map_lut(proposed, active_bank_version=1, config=config),
    )
    return images, initial, proposed


def _phase_sweep(cadence: ThreeTimescaleCadence) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for policy in ("first_influenced_window", "first_full_post_change_window"):
        records = cadence.phase_sweep(evidence_policy=policy)  # type: ignore[arg-type]
        for phase, record in enumerate(records):
            row = record.to_dict()
            row["onset_phase_index"] = phase
            rows.append(row)
        lags = np.asarray([record.total_lag_cycles for record in records], dtype=np.float64)
        evidence = np.asarray(
            [record.evidence_wait_cycles for record in records], dtype=np.float64
        )
        summary[policy] = {
            "phase_count": len(records),
            "min_lag_cycles": int(np.min(lags)),
            "median_lag_cycles": float(np.median(lags)),
            "mean_lag_cycles": float(np.mean(lags)),
            "max_lag_cycles": int(np.max(lags)),
            "min_lag_us": float(np.min(lags) * cadence.config.t_fast_us),
            "median_lag_us": float(np.median(lags) * cadence.config.t_fast_us),
            "max_lag_us": float(np.max(lags) * cadence.config.t_fast_us),
            "min_evidence_wait_cycles": int(np.min(evidence)),
            "max_evidence_wait_cycles": int(np.max(evidence)),
            "unique_lag_count": int(np.unique(lags).size),
        }
    return rows, summary


def _run_reference_trace(
    scheduler_config: SchedulerConfig,
    cadence: ThreeTimescaleCadence,
) -> dict[str, Any]:
    images, initial, proposed = _parameter_images()
    image_by_version = {image.active_bank_version: image for image in images}
    bank = ParamBank(initial_params=initial)

    def slow_update(_window: object, _active: object) -> DecoderRuntimeParams:
        result = proposed.copy()
        result.metadata["source"] = "t4.3.1_deterministic_trace"
        result.metadata["change_epoch"] = TRACE_ONSET_EPOCH
        return result

    actual_config = SchedulerConfig(
        t_fast_us=scheduler_config.t_fast_us,
        window_size=scheduler_config.window_size,
        slow_update_period_us=scheduler_config.slow_update_period_us,
        window_stride=scheduler_config.resolved_window_stride,
        max_pending_windows=scheduler_config.max_pending_windows,
        commit_delay_cycles=scheduler_config.commit_delay_cycles,
        fast_path_budget_us=scheduler_config.fast_path_budget_us,
        slow_path_budget_us=scheduler_config.slow_path_budget_us,
        guard_cycles_after_commit=scheduler_config.guard_cycles_after_commit,
        window_deadline_us=scheduler_config.resolved_window_deadline_us,
    )
    scheduler = DualLoopScheduler(
        actual_config,
        param_bank=bank,
        latency_injector=_constant_injector(service_us=cadence.config.slow_service_us),
        slow_path_fn=slow_update,
    )
    fast_path = BitAccurateFastPath(
        images, max_parameter_age_cycles=cadence.config.max_parameter_age_cycles
    )
    expected = cadence.adaptation_schedule(TRACE_ONSET_EPOCH)
    boundary_epochs = set(range(1, 8))
    for center in (
        TRACE_ONSET_EPOCH,
        expected.window_end_epoch,
        expected.slow_finish_epoch,
        expected.commit_epoch,
    ):
        boundary_epochs.update(range(center - 1, center + 2))

    trace_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    scheduler_events: list[Any] = []
    fast_version_by_epoch: dict[int, int] = {}
    event_mode_by_epoch: dict[int, str] = {}

    def fast_callback(epoch: int, time_us: float, will_emit_window: bool) -> dict[str, Any] | None:
        version = bank.active_version
        image = image_by_version[version]
        source_epoch = epoch - cadence.config.event_register_cycles
        local_event = source_epoch == TRACE_ONSET_EPOCH
        if epoch < fast_path.contract.map_pipeline_cycles:
            fast_version_by_epoch[epoch] = version
            event_mode_by_epoch[epoch] = "pipeline_warmup"
            if epoch in boundary_epochs:
                trace_rows.append(
                    {
                        "record_type": "fast_pipeline_warmup",
                        "epoch": epoch,
                        "time_us": time_us,
                        "source_epoch": source_epoch,
                        "active_version": version,
                        "event_mode": "pipeline_warmup",
                        "map_accepted": 0,
                        "fault_mask": 0,
                        "window_id": "",
                        "commit_epoch": "",
                        "detail": "no_registered_MAP_output_yet",
                    }
                )
            return None
        code_input = FastPathCodeInput(
            cycle_index=epoch,
            syndrome_code=1 << (image.config.adc_bits - 1),
            syndrome_x="leakage" if local_event else "g",
            syndrome_z="g",
            quadrature_phase_bit=epoch % 2,
            expected_active_bank_version=version,
            reported_image_crc32=image.image_crc32,
            reported_image_sha256=image.image_sha256,
            parameter_age_code=min(epoch - bank.last_commit_epoch, (1 << 16) - 1),
            ood_score_code=0,
            observation_valid=True,
        )
        result = fast_path.step_codes(code_input)
        hardware = result.fallback_action.hardware_action
        fast_version_by_epoch[epoch] = version
        event_mode_by_epoch[epoch] = hardware.mode
        if epoch in boundary_epochs:
            trace_rows.append(
                {
                    "record_type": "fast_action",
                    "epoch": epoch,
                    "time_us": time_us,
                    "source_epoch": source_epoch,
                    "active_version": version,
                    "event_mode": hardware.mode,
                    "map_accepted": int(result.fallback_action.map_decision_accepted),
                    "fault_mask": result.fallback_action.fault_mask,
                    "window_id": "",
                    "commit_epoch": "",
                    "detail": "local_event_leakage" if local_event else "regular",
                }
            )
        if not will_emit_window:
            return None
        window_start = epoch - actual_config.window_size + 1
        post_change_samples = max(0, epoch - max(window_start, TRACE_ONSET_EPOCH) + 1)
        return {
            "change_epoch": TRACE_ONSET_EPOCH,
            "post_change_samples": post_change_samples,
            "first_influenced": post_change_samples >= 1,
            "active_version_at_emit": version,
        }

    n_cycles = expected.commit_epoch + 2
    for _ in range(n_cycles):
        events = scheduler.tick_with_fast_path(fast_path_fn=fast_callback)
        for event in events:
            scheduler_events.append(event)
            row = {
                "record_type": event.kind,
                "epoch": event.epoch_id,
                "time_us": event.time_us,
                "source_epoch": "",
                "active_version": bank.active_version,
                "event_mode": "",
                "map_accepted": "",
                "fault_mask": "",
                "window_id": event.details.get("window_id", ""),
                "commit_epoch": event.details.get("commit_epoch", ""),
                "detail": json.dumps(event.details, sort_keys=True, ensure_ascii=True),
            }
            all_event_rows.append(row)
            trace_rows.append(row)

    def event_epoch(kind: str, *, version: int | None = None) -> int:
        matches = [
            int(event.epoch_id)
            for event in scheduler_events
            if event.kind == kind
            and (version is None or int(event.details.get("version", -1)) == version)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {kind} event, found {matches}")
        return matches[0]

    observed = {
        "event_source_epoch": TRACE_ONSET_EPOCH,
        "event_action_epoch": TRACE_ONSET_EPOCH + cadence.config.event_register_cycles,
        "event_action_mode": event_mode_by_epoch[
            TRACE_ONSET_EPOCH + cadence.config.event_register_cycles
        ],
        "window_ready_epoch": event_epoch("window_ready"),
        "slow_start_epoch": event_epoch("slow_update_started"),
        "slow_finish_epoch": event_epoch("slow_update_finished"),
        "stage_epoch": event_epoch("params_staged"),
        "commit_epoch": event_epoch("commit_applied", version=1),
        "first_fast_use_version": fast_version_by_epoch[expected.first_use_epoch],
        "precommit_fast_version": fast_version_by_epoch[expected.first_use_epoch - 1],
        "final_active_version": bank.active_version,
        "final_active_bank": bank.active_bank_name,
        "fast_path_rows_executed": len(fast_path.history),
        "pipeline_warmup_cycles": fast_path.contract.map_pipeline_cycles - 1,
        "scheduler_rows_recorded": len(all_event_rows),
    }
    return {
        "expected": expected.to_dict(),
        "observed": observed,
        "trace_rows": trace_rows,
        "trace_row_sha256": _row_hash(trace_rows),
    }


def _gate(gate_id: str, description: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {
        "id": gate_id,
        "description": description,
        "passed": bool(passed),
        "evidence": evidence,
    }


def run_validation(
    *,
    json_path: Path = DEFAULT_JSON,
    sweep_csv_path: Path = DEFAULT_SWEEP_CSV,
    trace_csv_path: Path = DEFAULT_TRACE_CSV,
) -> dict[str, Any]:
    raw_config, scheduler_config, cadence = _production_contract()
    sweep_rows, lag_summary = _phase_sweep(cadence)
    trace_a = _run_reference_trace(scheduler_config, cadence)
    trace_b = _run_reference_trace(scheduler_config, cadence)
    recalibration = cadence.recalibration_schedule(
        2 * cadence.config.recalibration_period_cycles + 137
    )
    recalibration_rows = [item.to_dict() for item in recalibration]

    influenced = lag_summary["first_influenced_window"]
    full = lag_summary["first_full_post_change_window"]
    expected = trace_a["expected"]
    observed = trace_a["observed"]
    config_dict = asdict(cadence.config)
    gates = [
        _gate(
            "G01",
            "hardware_hil production cadence is loaded without shadow constants",
            config_dict["t_fast_us"] == raw_config["runtime"]["t_fast_us"]
            and config_dict["window_size"] == raw_config["runtime"]["window_size"]
            and config_dict["window_stride"] == raw_config["runtime"]["window_stride"]
            and config_dict["slow_update_period_us"]
            == raw_config["runtime"]["t_slow_update_ms"] * 1000.0,
            config_dict,
        ),
        _gate(
            "G02",
            "fast, event, window, slow, commit and minute ratios are integer and phase locked",
            cadence.config.slow_period_cycles == 4000
            and cadence.config.event_register_cycles == 1
            and cadence.config.slow_service_cycles == 199
            and cadence.config.max_parameter_age_cycles == 8192
            and cadence.config.recalibration_period_cycles == 12_000_000,
            {
                "slow_period_cycles": cadence.config.slow_period_cycles,
                "event_register_cycles": cadence.config.event_register_cycles,
                "service_cycles": cadence.config.slow_service_cycles,
                "max_parameter_age_cycles": cadence.config.max_parameter_age_cycles,
                "minute_cycles": cadence.config.recalibration_period_cycles,
            },
        ),
        _gate(
            "G03",
            "every onset phase is enumerated under both evidence policies",
            len(sweep_rows) == 2 * cadence.config.window_stride
            and influenced["unique_lag_count"] == cadence.config.window_stride
            and full["unique_lag_count"] == cadence.config.window_stride,
            {"rows": len(sweep_rows), "summary": lag_summary},
        ),
        _gate(
            "G04",
            "optimistic first-influenced lag range includes evidence, service and commit",
            influenced["min_lag_cycles"] == 200
            and influenced["max_lag_cycles"] == 4199
            and influenced["median_lag_cycles"] == 2199.5,
            influenced,
        ),
        _gate(
            "G05",
            "full-post-change evidence lag is reported separately",
            full["min_lag_cycles"] == 2247
            and full["max_lag_cycles"] == 6246
            and full["median_lag_cycles"] == 4246.5,
            full,
        ),
        _gate(
            "G06",
            "local urgent event reaction remains one cycle and independent of host lag",
            all(row["event_lag_cycles"] == 1 for row in sweep_rows)
            and observed["event_action_epoch"] - observed["event_source_epoch"] == 1,
            {
                "event_lag_us": cadence.config.t_fast_us,
                "trace_mode": observed["event_action_mode"],
            },
        ),
        _gate(
            "G07",
            "real scheduler window/start/finish/stage/commit epochs equal analytic decomposition",
            observed["window_ready_epoch"] == expected["window_end_epoch"]
            and observed["slow_start_epoch"] == expected["slow_start_epoch"]
            and observed["slow_finish_epoch"] == expected["slow_finish_epoch"]
            and observed["stage_epoch"] == expected["stage_epoch"]
            and observed["commit_epoch"] == expected["commit_epoch"],
            {"expected": expected, "observed": observed},
        ),
        _gate(
            "G08",
            "commit precedes the same-cycle fast callback and the prior cycle keeps old version",
            observed["precommit_fast_version"] == 0
            and observed["first_fast_use_version"] == 1
            and observed["final_active_version"] == 1,
            observed,
        ),
        _gate(
            "G09",
            "T4.2 integer MAP-health-event-frame path executes every post-warmup scheduler cycle",
            observed["fast_path_rows_executed"]
            == expected["commit_epoch"] + 2 - observed["pipeline_warmup_cycles"]
            and observed["event_action_mode"] == "hold",
            observed,
        ),
        _gate(
            "G10",
            "independent recompilation and replay are byte-stable at the trace-row level",
            trace_a["trace_row_sha256"] == trace_b["trace_row_sha256"]
            and trace_a["observed"] == trace_b["observed"],
            {
                "trace_a": trace_a["trace_row_sha256"],
                "trace_b": trace_b["trace_row_sha256"],
            },
        ),
        _gate(
            "G11",
            "minute and end-of-run recalibration due signals are explicit and boundary aligned",
            [row["epoch"] for row in recalibration_rows]
            == [12_000_000, 24_000_000, 24_000_137]
            and all(
                row["epoch"] % cadence.config.slow_period_cycles == 0
                for row in recalibration_rows[:-1]
            )
            and recalibration_rows[-1]["kinds"] == ["end_of_run"],
            recalibration_rows,
        ),
        _gate(
            "G12",
            "lag components close exactly for every phase and no queue wait is hidden",
            all(
                row["total_lag_cycles"]
                == row["evidence_wait_cycles"]
                + row["queue_wait_cycles"]
                + row["service_cycles"]
                + row["commit_wait_cycles"]
                + row["first_use_wait_cycles"]
                and row["queue_wait_cycles"] == 0
                for row in sweep_rows
            ),
            {"rows_checked": len(sweep_rows)},
        ),
        _gate(
            "G13",
            "window semantics expose valid-sample content and emission interval separately",
            cadence.config.window_content_us == 10_240.0
            and cadence.config.slow_update_period_us == 20_000.0
            and cadence.config.window_size < cadence.config.window_stride,
            {
                "window_content_us": cadence.config.window_content_us,
                "window_emission_interval_us": cadence.config.slow_update_period_us,
            },
        ),
        _gate(
            "G14",
            "evidence is labelled software/config reference rather than RTL or board measurement",
            cadence.config.model_scope == MODEL_SCOPE
            and "not_rtl_or_board" in cadence.config.model_scope,
            cadence.config.model_scope,
        ),
    ]
    if not all(gate["passed"] for gate in gates):
        failed = [gate["id"] for gate in gates if not gate["passed"]]
        raise RuntimeError(f"T4.3.1 validation gates failed: {failed}")

    _write_csv(sweep_rows, sweep_csv_path)
    _write_csv(trace_a["trace_rows"], trace_csv_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "T4.3.1",
        "snapshot_date": "2026-07-15",
        "scope": MODEL_SCOPE,
        "configuration": config_dict,
        "cadence_definition": {
            "fast": "one T4.2 integer action per 5 us configured epoch, II=1",
            "event": "observed local event registered once and acted at the next fast boundary",
            "health_window": "2048 valid-sample snapshot emitted every 4000 fast cycles",
            "slow": "one eligible window job every 20 ms; reference mean service 995 us",
            "commit": "stage on completion; switch at next boundary before fast callback",
            "recalibration": "host due signal every 60 s and at explicit end-of-run; never direct active-bank mutation",
        },
        "scheduler_operation_order": [
            "increment epoch",
            "sample fast budget",
            "commit ready bank",
            "finish and stage slow job",
            "execute T4.2 fast callback",
            "emit window",
            "start next slow job",
        ],
        "adaptation_lag_definition": {
            "local_event": "event source to registered local action",
            "host_first_influenced": "drift onset to first fast action using a bank derived from any post-onset sample",
            "host_full_post_change": "drift onset to first fast action using a bank derived from a wholly post-onset window",
            "components": [
                "evidence_wait",
                "queue_wait",
                "slow_service",
                "stage_to_commit",
                "commit_to_first_use",
            ],
        },
        "lag_summary": lag_summary,
        "reference_execution_trace": {
            "onset_epoch": TRACE_ONSET_EPOCH,
            "expected": expected,
            "observed": observed,
            "trace_row_count": len(trace_a["trace_rows"]),
            "trace_row_sha256": trace_a["trace_row_sha256"],
        },
        "recalibration_schedule_example": recalibration_rows,
        "source_data": {
            "phase_sweep_csv": _display_path(sweep_csv_path),
            "phase_sweep_rows": len(sweep_rows),
            "phase_sweep_sha256": _sha256_file(sweep_csv_path),
            "trace_csv": _display_path(trace_csv_path),
            "trace_rows": len(trace_a["trace_rows"]),
            "trace_sha256": _sha256_file(trace_csv_path),
        },
        "provenance": {
            "config_path": "cnn_fpga/config/hardware_hil.yaml",
            "config_sha256": _sha256_file(CONFIG_PATH),
            "implementation_sha256": _implementation_sha256(),
            "deterministic_trace_replay_sha256": trace_a["trace_row_sha256"],
        },
        "gates": gates,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--trace-csv", type=Path, default=DEFAULT_TRACE_CSV)
    args = parser.parse_args()
    result = run_validation(
        json_path=args.json,
        sweep_csv_path=args.sweep_csv,
        trace_csv_path=args.trace_csv,
    )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "gates_passed": sum(gate["passed"] for gate in result["gates"]),
                "gates_total": len(result["gates"]),
                "lag_summary": result["lag_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
