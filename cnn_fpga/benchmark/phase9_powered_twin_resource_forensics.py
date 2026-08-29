"""Read-only reconstruction of a terminal T04 resource-preflight failure.

The V5c supervisor predated the late-failure ``completed_stage_evidence``
field.  Its immutable receipts, continuous resource samples and stage wall
timestamps are nevertheless sufficient to reconstruct a conservative
outcome-blind projection.  This module never opens formal artifacts or seed
addresses, never writes into a run directory and never issues a scientific
verdict.

Receipt publication time is used as the worker completion witness.  Stage
entry precedes process spawn, so the resulting worker wall is a conservative
upper bound (spawn latency is included).  The statistics stage is likewise
bounded by adjacent stage timestamps.  Because the historical physicality
sub-timing was not persisted, a one-second positive floor is added on top of
that already conservative stage bound; this can only make the wall decision
harder to pass.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    EXPECTED_CLAIM_FIELDS,
    build_cell_plan,
)
from cnn_fpga.benchmark.phase9_powered_twin_preflight import (
    _receipt_metrics,
    stratified_projection,
)


SCHEMA = "PHASE9-T04-RESOURCE-FAILURE-FORENSICS-V1"
EXPECTED_PROFILE_INDICES = (388, 389, 403, 478, 480, 482, 484, 507)
PROFILE_STAGE = {
    478: "formal_lpt_four_worker_peak",
    480: "formal_lpt_four_worker_peak",
    482: "formal_lpt_four_worker_peak",
    484: "formal_lpt_four_worker_peak",
    388: "representative_four_worker_profiles",
    389: "representative_four_worker_profiles",
    403: "representative_four_worker_profiles",
    507: "representative_four_worker_profiles",
}
ZERO_SHA256 = "0" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _verified_hashed_record(
    value: Mapping[str, Any],
    *,
    hash_field: str,
) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop(hash_field, None)
    if not isinstance(claimed, str) or claimed != _sha(unsigned):
        raise RuntimeError(f"invalid {hash_field}")


def _read_samples(
    path: Path,
    *,
    wall_anchor: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    first_by_stage: dict[str, dict[str, Any]] = {}
    last_by_stage: dict[str, dict[str, Any]] = {}
    count_by_stage: dict[str, int] = defaultdict(int)
    peak_by_stage_and_children: dict[str, dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    peak_child_rss = 0
    previous = ZERO_SHA256
    with path.open("r", encoding="utf-8") as handle:
        for expected_sequence, line in enumerate(handle):
            record = json.loads(line)
            if int(record.get("sequence", -1)) != expected_sequence:
                raise RuntimeError("resource sample sequence drift")
            if record.get("previous_sample_sha256") != previous:
                raise RuntimeError("resource sample hash-chain drift")
            _verified_hashed_record(record, hash_field="sample_sha256")
            previous = str(record["sample_sha256"])
            if "wall_time_ns" not in record:
                if wall_anchor is None:
                    raise RuntimeError(
                        "samples lack wall time and no heartbeat anchor was supplied"
                    )
                process_creation_time = wall_anchor.get("process_creation_time")
                if (
                    isinstance(process_creation_time, bool)
                    or not isinstance(process_creation_time, (int, float))
                    or not math.isfinite(float(process_creation_time))
                ):
                    raise RuntimeError("heartbeat process-creation anchor is invalid")
                record["forensic_wall_time_ns"] = int(
                    round(float(process_creation_time) * 1.0e9)
                    + round(
                        float(record["monotonic_seconds"]) * 1.0e9
                    )
                )
            stage = str(record["stage"])
            first_by_stage.setdefault(stage, record)
            last_by_stage[stage] = record
            count_by_stage[stage] += 1
            live = int(record["live_child_count"])
            peak_by_stage_and_children[stage][live] = max(
                peak_by_stage_and_children[stage][live],
                int(record["aggregate_rss_bytes"]),
            )
            child_rss = record.get("child_rss_bytes", {})
            if isinstance(child_rss, Mapping):
                observed_child_rss = [
                    int(value) for value in child_rss.values()
                ]
                if observed_child_rss:
                    peak_child_rss = max(
                        peak_child_rss,
                        max(observed_child_rss),
                    )
            records.append(record)
    if not records:
        raise RuntimeError("resource sample evidence is empty")
    summary = {
        "sample_count": len(records),
        "sample_chain_tip_sha256": previous,
        "first_by_stage": first_by_stage,
        "last_by_stage": last_by_stage,
        "count_by_stage": dict(count_by_stage),
        "peak_aggregate_rss_by_stage_and_live_children": {
            stage: {str(count): value for count, value in sorted(peaks.items())}
            for stage, peaks in sorted(peak_by_stage_and_children.items())
        },
        "peak_individual_child_rss_bytes": peak_child_rss,
        "maximum_observed_live_children": max(
            int(record["live_child_count"]) for record in records
        ),
        "peak_aggregate_rss_bytes": max(
            int(record["aggregate_rss_bytes"]) for record in records
        ),
    }
    return records, summary


def _stage_wall_ns(record: Mapping[str, Any]) -> int:
    wall = record.get("wall_time_ns", record.get("forensic_wall_time_ns"))
    if isinstance(wall, bool) or not isinstance(wall, int) or wall <= 0:
        raise RuntimeError("resource sample wall_time_ns is invalid")
    return wall


def _duration_seconds(start: Mapping[str, Any], stop: Mapping[str, Any]) -> float:
    duration = (_stage_wall_ns(stop) - _stage_wall_ns(start)) / 1.0e9
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError("nonpositive forensic stage duration")
    return duration


def _read_receipts(
    receipt_root: Path,
    *,
    stage_first: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    observed_indices: list[int] = []
    for path in sorted(receipt_root.glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        _verified_hashed_record(receipt, hash_field="receipt_sha256")
        plan_index = int(receipt["cell"]["plan_index"])
        observed_indices.append(plan_index)
        stage = PROFILE_STAGE.get(plan_index)
        if stage is None or stage not in stage_first:
            raise RuntimeError("unexpected resource-profile receipt")
        stage_start_ns = _stage_wall_ns(stage_first[stage])
        completion_ns = int(path.stat().st_mtime_ns)
        wall_seconds = (completion_ns - stage_start_ns) / 1.0e9
        if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
            raise RuntimeError("receipt completion precedes stage entry")
        result = {
            "receipt": receipt,
            "pid": 0,
            "wall_seconds": wall_seconds,
        }
        measurement = _receipt_metrics(result)
        measurement["wall_witness"] = {
            "method": "receipt_mtime_minus_stage_entry_wall_time",
            "stage": stage,
            "stage_entry_wall_time_ns": stage_start_ns,
            "receipt_mtime_ns": completion_ns,
            "spawn_latency_included": True,
            "conservative_upper_bound": True,
        }
        measurements.append(measurement)
    if tuple(sorted(observed_indices)) != EXPECTED_PROFILE_INDICES:
        raise RuntimeError("resource-profile receipt set is not the frozen eight")
    return sorted(measurements, key=lambda item: int(item["plan_index"]))


def ideal_lpt_curve(
    cell_projections: Sequence[Mapping[str, Any]],
    *,
    worker_counts: Iterable[int],
    fixed_overhead_seconds: float,
    maximum_wall_seconds: float,
) -> list[dict[str, Any]]:
    """Compute an ideal no-contention LPT lower-bound curve.

    The curve is diagnostic only.  It deliberately does not claim that CPU,
    memory, storage or scheduler semantics remain valid at higher concurrency.
    """

    curve: list[dict[str, Any]] = []
    for workers in worker_counts:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("worker count must be a positive integer")
        loads = [0.0] * workers
        for item in sorted(
            cell_projections,
            key=lambda value: (
                -float(value["projected_wall_seconds"]),
                int(value["plan_index"]),
            ),
        ):
            target = min(range(workers), key=lambda index: (loads[index], index))
            loads[target] += float(item["projected_wall_seconds"])
        raw_wall = max(loads, default=0.0)
        total_wall = raw_wall + float(fixed_overhead_seconds)
        curve.append(
            {
                "workers": workers,
                "ideal_raw_lpt_wall_seconds": raw_wall,
                "ideal_total_wall_seconds": total_wall,
                "ideal_total_wall_days": total_wall / 86400.0,
                "wall_gate_pass_if_no_contention": total_wall
                <= float(maximum_wall_seconds),
                "worker_load_seconds": loads,
            }
        )
    return curve


def reconstruct(
    repository_root: Path,
    run_directory: Path,
    *,
    physicality_positive_floor_seconds: float = 1.0,
) -> dict[str, Any]:
    root = repository_root.resolve()
    run = run_directory.resolve()
    if run.parent != (root / "runs").resolve():
        raise RuntimeError("run directory is outside the repository runs root")
    if (run / "owner.lock").exists():
        raise RuntimeError("refusing forensic reconstruction of an active run")
    if list(run.glob("*pass*.json")) or list(run.glob("*PASS*.json")):
        raise RuntimeError("terminal resource PASS conflicts with failure reconstruction")
    failure_path = run / "resource_preflight_failed.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    _verified_hashed_record(failure, hash_field="analysis_sha256")
    if (
        failure.get("verdict") != "INCOMPLETE_RESOURCE_FAIL_CLOSED"
        or failure.get("error") != "resource gates failed: wall"
    ):
        raise RuntimeError("run is not the expected wall fail-closed terminal")
    claim_boundary = failure.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping) or any(
        claim_boundary.get(name) is not None for name in EXPECTED_CLAIM_FIELDS
    ):
        raise RuntimeError("terminal claim boundary is not all null")

    config_path = root / "configs/phase9/t_risk_20260728_04_powered_twin_qualification.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_binding = _sha_file(config_path)
    if config_binding[1] != failure.get("config_sha256"):
        raise RuntimeError("live frozen config does not match the failed run")
    cells = build_cell_plan(config)
    if len(cells) != 518 or sum(cell.expected_rows for cell in cells) != 2_085_888:
        raise RuntimeError("formal denominator drift")

    samples_path = run / "resource_samples.jsonl"
    heartbeat_path = run / "heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    _verified_hashed_record(heartbeat, hash_field="heartbeat_sha256")
    _records, samples = _read_samples(samples_path, wall_anchor=heartbeat)
    stage_first = samples["first_by_stage"]
    stage_last = samples["last_by_stage"]
    required_stages = {
        "formal_lpt_four_worker_peak",
        "representative_four_worker_profiles",
        "joint_maxt_3037x199",
        "inventory_finalize_no_copy",
    }
    if not required_stages.issubset(stage_first):
        raise RuntimeError("resource sample stages are incomplete")
    measurements = _read_receipts(
        run / "receipts",
        stage_first=stage_first,
    )
    statistics_wall = _duration_seconds(
        stage_first["joint_maxt_3037x199"],
        stage_last["joint_maxt_3037x199"],
    )
    inventory_wall = _duration_seconds(
        stage_first["inventory_finalize_no_copy"],
        stage_last["inventory_finalize_no_copy"],
    )
    if (
        not math.isfinite(physicality_positive_floor_seconds)
        or physicality_positive_floor_seconds <= 0.0
    ):
        raise ValueError("physicality positive floor must be positive")
    inventory = json.loads((run / "inventory.json").read_text(encoding="utf-8"))
    projection = stratified_projection(
        config,
        cells,
        measurements,
        stats_wall_seconds=statistics_wall,
        retained_density_physicality_wall_seconds=physicality_positive_floor_seconds,
        inventory_finalize_wall_seconds=inventory_wall,
        inventory_profile_object_bytes=int(inventory["totals"]["object_bytes_unique"]),
        inventory_profile_receipt_count=int(inventory["receipt_count"]),
    )
    fixed_overhead = (
        float(projection["statistics_wall_seconds"])
        + float(projection["retained_density_physicality_serial_wall_seconds"])
        + float(projection["projected_inventory_finalize_wall_seconds"])
    )
    wall_limit = float(config["resource_contract"]["maximum_wall_seconds"])
    curve = ideal_lpt_curve(
        projection["cell_projections"],
        worker_counts=range(4, 17),
        fixed_overhead_seconds=fixed_overhead,
        maximum_wall_seconds=wall_limit,
    )
    passing = [item["workers"] for item in curve if item["wall_gate_pass_if_no_contention"]]
    top_cells = sorted(
        projection["cell_projections"],
        key=lambda item: (-float(item["projected_wall_seconds"]), int(item["plan_index"])),
    )[:20]
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task_id": "T-RISK-20260830-01",
        "source_run_id": failure["run_id"],
        "mode": "read_only_outcome_blind_resource_forensics",
        "formal_outcomes_accessed": False,
        "formal_seed_addresses_accessed": False,
        "scientific_verdict": None,
        "qualified_claim": None,
        "claim_boundary": {name: None for name in EXPECTED_CLAIM_FIELDS},
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "bytes": config_binding[0],
                "sha256": config_binding[1],
            },
            "failure": {
                "path": failure_path.relative_to(root).as_posix(),
                "bytes": _sha_file(failure_path)[0],
                "sha256": _sha_file(failure_path)[1],
            },
            "samples": {
                "path": samples_path.relative_to(root).as_posix(),
                "bytes": _sha_file(samples_path)[0],
                "sha256": _sha_file(samples_path)[1],
                "chain_tip_sha256": samples["sample_chain_tip_sha256"],
            },
            "heartbeat": {
                "path": heartbeat_path.relative_to(root).as_posix(),
                "bytes": _sha_file(heartbeat_path)[0],
                "sha256": _sha_file(heartbeat_path)[1],
                "heartbeat_sha256": heartbeat["heartbeat_sha256"],
                "used_as_process_creation_anchor": True,
            },
        },
        "terminal_checks": {
            "owner_lock_absent": True,
            "resource_pass_absent": True,
            "terminal_wall_failure": True,
            "claims_all_null": True,
            "receipt_count": len(measurements),
            "sample_count": samples["sample_count"],
        },
        "timing_reconstruction_contract": {
            "worker_wall_method": "receipt_mtime_minus_stage_entry_wall_time",
            "worker_spawn_latency_included": True,
            "statistics_method": "statistics_stage_first_to_last_sample_wall_delta",
            "inventory_method": "inventory_stage_first_to_last_sample_wall_delta",
            "physicality_positive_floor_seconds": physicality_positive_floor_seconds,
            "historical_subtiming_exactly_recoverable": False,
            "direction": "conservative_upper_bound_for_wall_rejection",
        },
        "profile_measurements": measurements,
        "sampling_resource_attribution": {
            key: samples[key]
            for key in (
                "maximum_observed_live_children",
                "peak_aggregate_rss_bytes",
                "peak_individual_child_rss_bytes",
                "peak_aggregate_rss_by_stage_and_live_children",
                "count_by_stage",
            )
        },
        "projection": projection,
        "ideal_no_contention_concurrency_curve": curve,
        "minimum_ideal_workers_passing_wall_gate": min(passing) if passing else None,
        "concurrency_inference_boundary": {
            "scheduler_change_authorized": False,
            "rss_at_higher_concurrency_proven": False,
            "cpu_scaling_at_higher_concurrency_proven": False,
            "disk_scaling_at_higher_concurrency_proven": False,
            "higher_concurrency_release": False,
        },
        "top_projected_wall_cells": top_cells,
        "resource_decision": {
            "maximum_wall_seconds": wall_limit,
            "projected_wall_seconds": projection[
                "projected_formal_wall_seconds_at_frozen_concurrency"
            ],
            "wall_ratio": projection[
                "projected_formal_wall_seconds_at_frozen_concurrency"
            ]
            / wall_limit,
            "wall_pass": False,
            "projected_artifact_bytes": projection[
                "projected_formal_artifact_bytes"
            ],
            "artifact_pass": int(projection["projected_formal_artifact_bytes"])
            <= int(config["resource_contract"]["maximum_artifact_bytes"]),
            "full_run_release": False,
        },
    }
    report["analysis_sha256"] = _sha(report)
    return report


def compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a review-sized view while retaining the full projection hash."""

    projection = report["projection"]
    compact = {
        key: report[key]
        for key in (
            "schema_version",
            "task_id",
            "source_run_id",
            "mode",
            "formal_outcomes_accessed",
            "formal_seed_addresses_accessed",
            "scientific_verdict",
            "qualified_claim",
            "claim_boundary",
            "source_bindings",
            "terminal_checks",
            "timing_reconstruction_contract",
            "sampling_resource_attribution",
            "ideal_no_contention_concurrency_curve",
            "minimum_ideal_workers_passing_wall_gate",
            "concurrency_inference_boundary",
            "top_projected_wall_cells",
            "resource_decision",
            "analysis_sha256",
        )
    }
    compact["projection_summary"] = {
        key: projection[key]
        for key in (
            "projection_sha256",
            "projected_formal_artifact_bytes",
            "projected_formal_worker_wall_seconds",
            "projected_raw_lpt_worker_load_seconds",
            "projected_raw_lpt_wall_seconds",
            "projected_formal_wall_seconds_at_frozen_concurrency",
            "frozen_concurrency",
            "statistics_wall_seconds",
            "retained_density_physicality_serial_wall_seconds",
            "inventory_profile_finalize_wall_seconds",
            "inventory_finalize_projection_scale",
            "projected_inventory_finalize_wall_seconds",
            "layers",
        )
    }
    return compact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="emit all 518 cell projections instead of the compact review view",
    )
    args = parser.parse_args(argv)
    report = reconstruct(args.repository_root, args.run_directory)
    print(
        json.dumps(
            report if args.full else compact_report(report),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
