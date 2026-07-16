"""T4.3.3 multi-scenario closed-loop stability and fault-recovery validation."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import textwrap
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.atomic_parameter_bank import serialize_parameter_image
from cnn_fpga.runtime.closed_loop_fault_recovery import (
    MODEL_SCOPE,
    ClosedLoopCycleInput,
    ClosedLoopFaultRecoverySupervisor,
    ClosedLoopRecoveryConfig,
    parameter_image_semantics_sha256,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "t4_3_3_closed_loop_fault_recovery_validation.json"
DEFAULT_CSV = ROOT / "docs" / "t4_3_3_closed_loop_fault_recovery_source_data.csv"
SCHEMA_VERSION = "t4.3.3-closed-loop-fault-recovery-validation-v1"
SCENARIOS = (
    "nominal_drift",
    "burst",
    "leakage_reset",
    "host_timeout",
    "communication_pause_ack_loss",
    "corrupt_transfer",
    "update_race",
    "post_commit_guard_republish",
)
DEFINED_ACTIONS = frozenset({"use_validated_map", "frame_hold", "reset_request"})
DEFINED_MODES = frozenset(
    {"normal", "x_recovery", "z_recovery", "hold", "reset_request", "fallback"}
)


@dataclass(frozen=True)
class FaultCampaignConfig:
    n_cycles: int = 24_000
    seeds: tuple[int, ...] = (431, 433, 439, 443)
    initial_stage_epoch: int = 6_000
    initial_apply_epoch: int = 6_200
    burst_start_epoch: int = 11_000
    burst_cycles: int = 64
    host_last_heartbeat_epoch: int = 7_000
    host_resume_epoch: int = 17_000
    communication_pause_start_epoch: int = 6_100
    communication_resume_epoch: int = 7_000
    recovery_apply_epoch: int = 17_200

    def __post_init__(self) -> None:
        if self.n_cycles < 20_000:
            raise ValueError("production fault campaign requires at least 20,000 cycles")
        if len(self.seeds) < 4 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("production fault campaign requires four unique seeds")
        if not (
            5 < self.initial_stage_epoch < self.initial_apply_epoch < self.n_cycles
            and self.burst_start_epoch + self.burst_cycles < self.n_cycles
            and self.host_last_heartbeat_epoch < self.host_resume_epoch < self.recovery_apply_epoch
            and self.recovery_apply_epoch < self.n_cycles
        ):
            raise ValueError("fault epochs must be ordered and inside the campaign")


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/closed_loop_fault_recovery.py",
        "cnn_fpga/runtime/atomic_parameter_bank.py",
        "cnn_fpga/runtime/fast_path_fixed_point.py",
        "cnn_fpga/runtime/conservative_fallback.py",
        "cnn_fpga/runtime/experimental_event_fsm.py",
        "cnn_fpga/benchmark/closed_loop_fault_recovery_validation.py",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(set(row) for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _images_for_scenario(scenario: str):
    config = ParametricMAPLUTConfig()
    profiles = registered_parameter_profiles(config)
    recovery_profile = 0 if scenario == "post_commit_guard_republish" else 1
    profile_ids = (0, 1, recovery_profile, recovery_profile)
    return tuple(
        compile_parametric_map_lut(
            profiles[profile][0], active_bank_version=version, config=config
        )
        for version, profile in enumerate(profile_ids)
    )


def _drift_code(epoch: int, seed: int) -> int:
    phase = seed % 97
    value = (
        512.0
        + 265.0 * math.sin(2.0 * math.pi * (epoch + phase) / 5_003.0)
        + 71.0 * math.sin(2.0 * math.pi * (epoch + 3 * phase) / 997.0)
    )
    return max(0, min(1023, int(round(value))))


def _prime_candidate(supervisor: ClosedLoopFaultRecoverySupervisor) -> None:
    supervisor.observe_selection(window_id=1, selection_key="adaptive", eligible=True)
    supervisor.observe_selection(window_id=2, selection_key="adaptive", eligible=True)


def _stage_initial_update(
    supervisor: ClosedLoopFaultRecoverySupervisor,
    images: tuple[Any, ...],
    *,
    scenario: str,
    config: FaultCampaignConfig,
    seed: int,
) -> list[dict[str, Any]]:
    _prime_candidate(supervisor)
    rows: list[dict[str, Any]] = []
    if scenario == "update_race":
        def attempt(index: int):
            return supervisor.submit_update(
                images[1],
                transaction_id=f"race-{seed}-{index}",
                selection_key="adaptive",
                source_window_id=2,
                created_epoch=config.initial_stage_epoch,
                apply_epoch=config.initial_apply_epoch,
                reverse_chunks=bool(index),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(attempt, range(2)))
    elif scenario == "corrupt_transfer":
        payload = bytearray(serialize_parameter_image(images[1]))
        payload[(seed * 17) % len(payload)] ^= 1
        attempts = [
            supervisor.submit_update(
                images[1],
                transaction_id=f"corrupt-{seed}",
                selection_key="adaptive",
                source_window_id=2,
                created_epoch=config.initial_stage_epoch,
                apply_epoch=config.initial_apply_epoch,
                payload_override=bytes(payload),
            )
        ]
    else:
        attempts = [
            supervisor.submit_update(
                images[1],
                transaction_id=f"candidate-{scenario}-{seed}",
                selection_key="adaptive",
                source_window_id=2,
                created_epoch=config.initial_stage_epoch,
                apply_epoch=config.initial_apply_epoch,
                reverse_chunks=bool(seed % 2),
            )
        ]
    for attempt in attempts:
        row = asdict(attempt)
        row.update(
            {
                "record_type": "update_attempt",
                "scenario": scenario,
                "seed": seed,
            }
        )
        rows.append(row)
    return rows


def _cycle_input(
    scenario: str,
    *,
    epoch: int,
    seed: int,
    config: FaultCampaignConfig,
    reset_ack: bool,
) -> ClosedLoopCycleInput:
    communication = not (
        scenario == "communication_pause_ack_loss"
        and config.communication_pause_start_epoch
        <= epoch
        < config.communication_resume_epoch
    )
    if scenario == "host_timeout":
        heartbeat_allowed = (
            epoch <= config.host_last_heartbeat_epoch or epoch >= config.host_resume_epoch
        )
    else:
        heartbeat_allowed = True
    heartbeat = communication and heartbeat_allowed and (
        epoch == 5 or epoch % 1_000 == 0 or epoch == config.host_resume_epoch
    )
    syndrome_x = "g"
    syndrome_z = "g"
    ood = 24
    deadline_ok = True
    integrity_ok = True
    burst_end = config.burst_start_epoch + config.burst_cycles
    if scenario == "burst" and config.burst_start_epoch <= epoch < burst_end:
        syndrome_x = "e"
        syndrome_z = "e" if (epoch + seed) % 3 == 0 else "g"
        ood = 224
        deadline_ok = False
    if scenario == "leakage_reset" and config.burst_start_epoch <= epoch < config.burst_start_epoch + 3:
        syndrome_x = "leakage"
    if scenario == "post_commit_guard_republish" and epoch in (
        config.initial_apply_epoch + 1,
        config.initial_apply_epoch + 2,
    ):
        integrity_ok = False
    safe_boundary = not (
        scenario == "nominal_drift" and epoch == config.initial_apply_epoch
    )
    return ClosedLoopCycleInput(
        epoch=epoch,
        syndrome_code=_drift_code(epoch, seed),
        syndrome_x=syndrome_x,
        syndrome_z=syndrome_z,
        quadrature_phase_bit=epoch % 2,
        ood_score_code=ood,
        host_heartbeat=heartbeat,
        communication_available=communication,
        safe_boundary=safe_boundary,
        reset_ack=reset_ack,
        deadline_ok=deadline_ok,
        reported_integrity_ok=integrity_ok,
    )


def _segment_key(record: Any) -> tuple[Any, ...]:
    return (
        record.active_version,
        record.health_status,
        record.action_mode,
        record.conservative_action,
        record.fault_mask,
        record.host_timed_out,
        record.communication_available,
        record.commit_status,
        record.readback_status,
        record.recovery_requested,
        record.recovery_reason,
    )


def _segment_row(
    scenario: str,
    seed: int,
    start: int,
    end: int,
    record: Any,
) -> dict[str, Any]:
    return {
        "record_type": "cycle_segment",
        "scenario": scenario,
        "seed": seed,
        "start_epoch": start,
        "end_epoch": end,
        "cycles": end - start + 1,
        "active_version": record.active_version,
        "active_bank": record.active_bank,
        "health_status": record.health_status,
        "action_mode": record.action_mode,
        "conservative_action": record.conservative_action,
        "fault_mask": record.fault_mask,
        "fault_flags": "|".join(record.fault_flags),
        "host_timed_out": int(record.host_timed_out),
        "communication_available": int(record.communication_available),
        "commit_status": record.commit_status,
        "commit_reason": record.commit_reason,
        "readback_status": record.readback_status,
        "recovery_requested": int(record.recovery_requested),
        "recovery_reason": record.recovery_reason,
        "active_semantics_sha256": record.active_semantics_sha256,
        "reason_trace": record.reason_trace,
    }


def _run_campaign(
    scenario: str,
    seed: int,
    config: FaultCampaignConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images = _images_for_scenario(scenario)
    supervisor = ClosedLoopFaultRecoverySupervisor(images)
    source_rows: list[dict[str, Any]] = []
    detail_hash = hashlib.sha256()
    fault_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    commit_epochs: list[int] = []
    commit_versions: list[int] = []
    versions: list[int] = []
    syndrome_codes: set[int] = set()
    undefined_actions = 0
    blocking_correction = 0
    frame_out_of_range = 0
    reset_request_cycles = 0
    fallback_cycles = 0
    awaiting_readback_cycles = 0
    ack_timeout_cycles = 0
    confirmed_readbacks = 0
    reset_ack_next = False
    current_segment_start = 5
    current_segment_record = None
    current_segment_key = None
    recovery_submit_done = False
    late_valid_submit_done = False
    next_window_id = 3
    if scenario == "host_timeout":
        refresh_stage_epochs: tuple[int, ...] = ()
    elif scenario == "post_commit_guard_republish":
        refresh_stage_epochs = (18_000,)
    else:
        refresh_stage_epochs = (14_000, 22_000)

    for epoch in range(5, config.n_cycles + 1):
        if epoch == config.initial_stage_epoch:
            source_rows.extend(
                _stage_initial_update(
                    supervisor,
                    images,
                    scenario=scenario,
                    config=config,
                    seed=seed,
                )
            )
        if scenario == "corrupt_transfer" and epoch == 8_000 and not late_valid_submit_done:
            supervisor.observe_selection(window_id=3, selection_key="adaptive", eligible=True)
            supervisor.observe_selection(window_id=4, selection_key="adaptive", eligible=True)
            attempt = supervisor.submit_update(
                images[1],
                transaction_id=f"valid-after-corrupt-{seed}",
                selection_key="adaptive",
                source_window_id=4,
                created_epoch=epoch,
                apply_epoch=8_200,
            )
            row = asdict(attempt)
            row.update({"record_type": "update_attempt", "scenario": scenario, "seed": seed})
            source_rows.append(row)
            late_valid_submit_done = True
            next_window_id = 5
        if (
            scenario == "post_commit_guard_republish"
            and epoch == 6_400
            and supervisor.recovery_requested
            and not recovery_submit_done
        ):
            attempt = supervisor.submit_lkg_republish(
                images[2],
                transaction_id=f"guard-lkg-{seed}",
                selection_key="guard-lkg",
                evidence_window_ids=(3, 4),
                created_epoch=epoch,
                apply_epoch=6_600,
            )
            row = asdict(attempt)
            row.update({"record_type": "update_attempt", "scenario": scenario, "seed": seed})
            source_rows.append(row)
            recovery_submit_done = True
            next_window_id = 5
        if (
            scenario == "host_timeout"
            and epoch == config.host_resume_epoch
            and supervisor.recovery_requested
            and not recovery_submit_done
        ):
            attempt = supervisor.submit_lkg_republish(
                images[2],
                transaction_id=f"stale-lkg-{seed}",
                selection_key="stale-lkg",
                evidence_window_ids=(3, 4),
                created_epoch=epoch,
                apply_epoch=config.recovery_apply_epoch,
            )
            row = asdict(attempt)
            row.update({"record_type": "update_attempt", "scenario": scenario, "seed": seed})
            source_rows.append(row)
            recovery_submit_done = True
            next_window_id = 5
        if epoch in refresh_stage_epochs:
            version = supervisor.bank.active_version + 1
            refresh = images[version]
            first_window = next_window_id
            second_window = next_window_id + 1
            key = f"cadence-refresh-v{version}"
            supervisor.observe_selection(
                window_id=first_window, selection_key=key, eligible=True
            )
            supervisor.observe_selection(
                window_id=second_window, selection_key=key, eligible=True
            )
            attempt = supervisor.submit_update(
                refresh,
                transaction_id=f"refresh-{scenario}-{seed}-v{version}",
                selection_key=key,
                source_window_id=second_window,
                created_epoch=epoch,
                apply_epoch=epoch + 200,
                purpose="candidate",
            )
            row = asdict(attempt)
            row.update(
                {"record_type": "update_attempt", "scenario": scenario, "seed": seed}
            )
            source_rows.append(row)
            next_window_id += 2

        cycle = _cycle_input(
            scenario,
            epoch=epoch,
            seed=seed,
            config=config,
            reset_ack=reset_ack_next,
        )
        record = supervisor.tick(cycle)
        reset_ack_next = record.reset_request
        encoded = json.dumps(
            record.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        detail_hash.update(encoded)
        syndrome_codes.add(cycle.syndrome_code)
        versions.append(record.active_version)
        action_counts[record.conservative_action] += 1
        fault_counts.update(record.fault_flags)
        reset_request_cycles += int(record.reset_request)
        fallback_cycles += int(record.conservative_action != "use_validated_map")
        awaiting_readback_cycles += int(record.readback_status == "awaiting_ack_readback")
        ack_timeout_cycles += int(record.readback_status == "ack_timeout_awaiting_readback")
        confirmed_readbacks += int(record.readback_status == "confirmed")
        if record.commit_status == "committed":
            commit_epochs.append(epoch)
            commit_versions.append(record.active_version)
        if (
            record.conservative_action not in DEFINED_ACTIONS
            or record.action_mode not in DEFINED_MODES
            or not record.action_reason
            or not record.reason_trace
            or not record.active_profile_id
        ):
            undefined_actions += 1
        blocking = set(record.fault_flags) - {"leakage_observed"}
        if blocking and record.correction_enable:
            blocking_correction += 1
        if not (
            0 <= record.phase_frame_x_code < 256
            and 0 <= record.phase_frame_z_code < 256
        ):
            frame_out_of_range += 1

        key = _segment_key(record)
        if current_segment_record is None:
            current_segment_record = record
            current_segment_key = key
            current_segment_start = epoch
        elif key != current_segment_key:
            source_rows.append(
                _segment_row(
                    scenario, seed, current_segment_start, epoch - 1, current_segment_record
                )
            )
            current_segment_record = record
            current_segment_key = key
            current_segment_start = epoch

    assert current_segment_record is not None
    source_rows.append(
        _segment_row(
            scenario,
            seed,
            current_segment_start,
            config.n_cycles,
            current_segment_record,
        )
    )
    attempts = [asdict(item) for item in supervisor.update_attempts]
    summary = {
        "scenario": scenario,
        "seed": seed,
        "cycles_executed": config.n_cycles - 4,
        "trace_sha256": detail_hash.hexdigest(),
        "syndrome_code_min": min(syndrome_codes),
        "syndrome_code_max": max(syndrome_codes),
        "syndrome_code_unique": len(syndrome_codes),
        "undefined_action_count": undefined_actions,
        "blocking_fault_with_correction_count": blocking_correction,
        "frame_out_of_range_count": frame_out_of_range,
        "active_version_monotonic": all(a <= b for a, b in zip(versions, versions[1:])),
        "active_versions": sorted(set(versions)),
        "commit_epochs": commit_epochs,
        "commit_versions": commit_versions,
        "fault_counts": dict(sorted(fault_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "reset_request_cycles": reset_request_cycles,
        "fallback_cycles": fallback_cycles,
        "awaiting_readback_cycles": awaiting_readback_cycles,
        "ack_timeout_cycles": ack_timeout_cycles,
        "confirmed_readbacks": confirmed_readbacks,
        "final_record": supervisor.records[-1].to_dict(),
        "update_attempts": attempts,
        "recovery_submit_done": recovery_submit_done,
        "final_lkg_semantics_sha256": supervisor.last_known_good_semantics_sha256,
        "v0_semantics_sha256": parameter_image_semantics_sha256(images[0]),
        "v1_semantics_sha256": parameter_image_semantics_sha256(images[1]),
        "v2_semantics_sha256": parameter_image_semantics_sha256(images[2]),
        "v3_semantics_sha256": parameter_image_semantics_sha256(images[3]),
    }
    source_rows.append(
        {
            "record_type": "run_summary",
            "scenario": scenario,
            "seed": seed,
            "cycles": summary["cycles_executed"],
            "active_version": summary["final_record"]["active_version"],
            "fault_flags": "|".join(summary["fault_counts"]),
            "commit_status": "|".join(map(str, commit_versions)),
            "readback_status": summary["final_record"]["readback_status"],
            "recovery_requested": int(summary["final_record"]["recovery_requested"]),
            "active_semantics_sha256": summary["final_record"]["active_semantics_sha256"],
            "reason_trace": summary["trace_sha256"],
        }
    )
    return summary, source_rows


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
    csv_path: Path = DEFAULT_CSV,
    config: FaultCampaignConfig | None = None,
) -> dict[str, Any]:
    actual = FaultCampaignConfig() if config is None else config
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in actual.seeds:
            summary, source = _run_campaign(scenario, seed, actual)
            results.append(summary)
            rows.extend(source)
    deterministic_a, _ = _run_campaign("post_commit_guard_republish", actual.seeds[0], actual)
    deterministic_b, _ = _run_campaign("post_commit_guard_republish", actual.seeds[0], actual)
    by_scenario = {
        scenario: [row for row in results if row["scenario"] == scenario]
        for scenario in SCENARIOS
    }
    all_cycles = sum(row["cycles_executed"] for row in results)
    runtime_tree = ast.parse(
        (ROOT / "cnn_fpga" / "runtime" / "closed_loop_fault_recovery.py").read_text(
            encoding="utf-8"
        )
    )
    forbidden_names = {"truth", "hidden_state", "target_params", "oracle"}
    loaded_names = {
        node.id for node in ast.walk(runtime_tree) if isinstance(node, ast.Name)
    }
    commit_source = ast.parse(
        textwrap.dedent(inspect.getsource(ClosedLoopFaultRecoverySupervisor.tick))
    )
    commit_loops = sum(
        isinstance(node, (ast.For, ast.While)) for node in ast.walk(commit_source)
    )
    guard = by_scenario["post_commit_guard_republish"]
    host = by_scenario["host_timeout"]
    pause = by_scenario["communication_pause_ack_loss"]
    corrupt = by_scenario["corrupt_transfer"]
    race = by_scenario["update_race"]
    gates = [
        _gate(
            "G01",
            "all eight required scenarios and four seeds execute the full production horizon",
            len(results) == len(SCENARIOS) * len(actual.seeds)
            and all(row["cycles_executed"] == actual.n_cycles - 4 for row in results),
            {"runs": len(results), "cycles": all_cycles, "scenarios": list(SCENARIOS)},
        ),
        _gate(
            "G02",
            "every cycle produces a defined action, mode, reason trace and profile",
            all(row["undefined_action_count"] == 0 for row in results),
            {"undefined": sum(row["undefined_action_count"] for row in results)},
        ),
        _gate(
            "G03",
            "blocking faults always inhibit correction and all frame words remain bounded",
            all(
                row["blocking_fault_with_correction_count"] == 0
                and row["frame_out_of_range_count"] == 0
                for row in results
            ),
            {
                "blocking_correction": sum(
                    row["blocking_fault_with_correction_count"] for row in results
                ),
                "frame_out_of_range": sum(row["frame_out_of_range_count"] for row in results),
            },
        ),
        _gate(
            "G04",
            "drift traverses a nontrivial ADC range without hidden truth or undefined action",
            all(
                row["syndrome_code_unique"] > 500
                and row["syndrome_code_max"] - row["syndrome_code_min"] > 500
                for row in by_scenario["nominal_drift"]
            )
            and not (loaded_names & forbidden_names),
            {
                "runs": [
                    {
                        "seed": row["seed"],
                        "min": row["syndrome_code_min"],
                        "max": row["syndrome_code_max"],
                        "unique": row["syndrome_code_unique"],
                    }
                    for row in by_scenario["nominal_drift"]
                ],
                "forbidden_loaded_names": sorted(loaded_names & forbidden_names),
            },
        ),
        _gate(
            "G05",
            "burst produces OOD/deadline fallback and returns to healthy defined action",
            all(
                row["fault_counts"].get("ood_score_exceeded", 0) > 0
                and row["fault_counts"].get("deadline_miss", 0) > 0
                and row["fallback_cycles"] > 0
                and row["final_record"]["health_status"] == "healthy"
                for row in by_scenario["burst"]
            ),
            by_scenario["burst"],
        ),
        _gate(
            "G06",
            "leakage invokes local reset request and recovers without host-dependent action",
            all(
                row["fault_counts"].get("leakage_observed", 0) >= 3
                and row["reset_request_cycles"] > 0
                and row["final_record"]["health_status"] == "healthy"
                for row in by_scenario["leakage_reset"]
            ),
            by_scenario["leakage_reset"],
        ),
        _gate(
            "G07",
            "host timeout and stale image enter fallback then monotonic LKG republish restores service",
            all(
                row["fault_counts"].get("deadline_miss", 0) > 0
                and row["fault_counts"].get("parameter_stale", 0) > 0
                and row["active_versions"] == [0, 1, 2]
                and row["final_record"]["active_semantics_sha256"]
                == row["v1_semantics_sha256"]
                and not row["final_record"]["recovery_requested"]
                and row["final_record"]["health_status"] == "healthy"
                for row in host
            ),
            host,
        ),
        _gate(
            "G08",
            "communication pause loses ack, times out as uncertain and later confirms by readback",
            all(
                row["awaiting_readback_cycles"] > 0
                and row["ack_timeout_cycles"] > 0
                and row["confirmed_readbacks"] > 0
                and row["final_record"]["awaiting_readback_version"] is None
                for row in pause
            ),
            pause,
        ),
        _gate(
            "G09",
            "every corrupt transfer is rejected and a later complete image alone may activate",
            all(
                any(
                    item["reason"] == "transfer_crc_mismatch" and not item["accepted"]
                    for item in row["update_attempts"]
                )
                and row["active_versions"] == [0, 1, 2, 3]
                and row["commit_versions"] == [1, 2, 3]
                for row in corrupt
            ),
            corrupt,
        ),
        _gate(
            "G10",
            "concurrent update race has exactly one staged winner and one explicit conflict",
            all(
                sum(
                    item["accepted"]
                    for item in row["update_attempts"]
                    if item["transaction_id"].startswith("race-")
                )
                == 1
                and sum(
                    item["reason"] in (
                        "writer_conflict_transfer_in_progress",
                        "writer_conflict_pending_commit",
                    )
                    for item in row["update_attempts"]
                    if item["transaction_id"].startswith("race-")
                )
                == 1
                and row["commit_versions"] == [1, 2, 3]
                for row in race
            ),
            race,
        ),
        _gate(
            "G11",
            "post-commit integrity guard republishes previous LKG contents under monotonic v2",
            all(
                row["fault_counts"].get("image_crc_mismatch", 0) >= 2
                and row["fault_counts"].get("image_sha256_mismatch", 0) >= 2
                and row["active_versions"] == [0, 1, 2, 3]
                and row["commit_versions"] == [1, 2, 3]
                and row["commit_epochs"][1] >= 10_200
                and row["final_record"]["active_semantics_sha256"]
                == row["v0_semantics_sha256"]
                and row["final_record"]["active_semantics_sha256"]
                != row["v1_semantics_sha256"]
                for row in guard
            ),
            guard,
        ),
        _gate(
            "G12",
            "active versions are monotonic in every run and no version rollback is used for recovery",
            all(row["active_version_monotonic"] for row in results),
            {"all_monotonic": all(row["active_version_monotonic"] for row in results)},
        ),
        _gate(
            "G13",
            "ack uncertainty blocks writes and every campaign closes pending readback/recovery state",
            all(
                row["final_record"]["awaiting_readback_version"] is None
                and not row["final_record"]["recovery_requested"]
                for row in results
            ),
            {
                "final_pending": sum(
                    row["final_record"]["awaiting_readback_version"] is not None
                    for row in results
                ),
                "final_recovery_requested": sum(
                    row["final_record"]["recovery_requested"] for row in results
                ),
            },
        ),
        _gate(
            "G14",
            "production 5us/8192/4000-cycle safety policies are preserved",
            ClosedLoopRecoveryConfig().fast_cycle_ns == 5_000
            and ClosedLoopRecoveryConfig().max_parameter_age_cycles == 8_192
            and ClosedLoopFaultRecoverySupervisor(_images_for_scenario("nominal_drift")).bank.config.min_residency_cycles
            == 4_000,
            asdict(ClosedLoopRecoveryConfig()),
        ),
        _gate(
            "G15",
            "online tick contains no payload mutation loop and uses observed/integrity inputs only",
            commit_loops == 0 and not (loaded_names & forbidden_names),
            {"tick_loop_nodes": commit_loops, "forbidden_loaded_names": sorted(loaded_names & forbidden_names)},
        ),
        _gate(
            "G16",
            "independent full guard campaign replays are deterministic",
            deterministic_a["trace_sha256"] == deterministic_b["trace_sha256"]
            and deterministic_a["commit_epochs"] == deterministic_b["commit_epochs"],
            {
                "run_a": deterministic_a["trace_sha256"],
                "run_b": deterministic_b["trace_sha256"],
                "commit_epochs": deterministic_a["commit_epochs"],
            },
        ),
        _gate(
            "G17",
            "evidence scope remains software fault recovery rather than RTL or board measurement",
            MODEL_SCOPE.endswith("not_rtl_or_board"),
            MODEL_SCOPE,
        ),
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, csv_path)
    source_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    result = {
        "task_id": "T4.3.3",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(gate["passed"] for gate in gates) else "FAIL",
        "scope": MODEL_SCOPE,
        "config": asdict(actual),
        "implementation_sha256": _implementation_sha256(),
        "source_data": {
            "path": csv_path.relative_to(ROOT).as_posix()
            if csv_path.is_relative_to(ROOT)
            else str(csv_path),
            "rows": len(rows),
            "sha256": source_sha,
            "row_payload_sha256": _row_hash(rows),
        },
        "summary": {
            "runs": len(results),
            "cycles_executed": all_cycles,
            "scenarios": list(SCENARIOS),
            "undefined_action_count": sum(row["undefined_action_count"] for row in results),
            "blocking_fault_with_correction_count": sum(
                row["blocking_fault_with_correction_count"] for row in results
            ),
            "frame_out_of_range_count": sum(row["frame_out_of_range_count"] for row in results),
            "per_run": results,
        },
        "gates": gates,
        "claim_boundary": {
            "allowed": "multi-scenario closed-loop software fault-recovery contract",
            "forbidden": [
                "RTL or CDC recovery proof",
                "FPGA or board fault recovery",
                "device-calibrated timeout, OOD, leakage, or reset efficacy",
                "physical logical-lifetime stability claim",
            ],
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    result = run_validation(json_path=args.json, csv_path=args.csv)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates_passed": sum(gate["passed"] for gate in result["gates"]),
                "gates_total": len(result["gates"]),
                "runs": result["summary"]["runs"],
                "cycles": result["summary"]["cycles_executed"],
                "source_rows": result["source_data"]["rows"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
