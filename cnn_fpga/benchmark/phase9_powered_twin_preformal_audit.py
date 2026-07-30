"""Outcome-blind final release gate for the T04 powered qualification.

The seal is created only after the V5 plan/seed/source contract, full-size resource
preflight, exact source snapshot and focused anti-simplification tests agree.
It releases one raw-evidence execution, not a scientific verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

from cnn_fpga.benchmark.phase9_immutable_object_store import (
    ATTEMPT_SCHEMA,
    INVENTORY_SCHEMA,
    ImmutableObjectStore,
)
from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    EXPECTED_CLAIM_FIELDS,
    TASK_ID,
    build_cell_plan,
    load_config,
    plan_payload,
    runtime_source_snapshot,
    seed_registry_payload,
)
from cnn_fpga.benchmark.phase9_powered_twin_plan import historical_seed_scan
from cnn_fpga.benchmark.phase9_powered_twin_preflight import (
    PASS_VERDICT,
    PREFLIGHT_SCHEMA,
    RAW_RUNNER_ID,
    RUNNER_ID as RESOURCE_RUNNER_ID,
    SAMPLING_SCHEMA,
    _object_tree_snapshot,
    assert_seed_firewall,
    audit_resource_profile_receipts,
    isolated_preflight_paths,
    profile_cells,
    resource_gate_decision,
    stratified_projection,
    validate_continuous_sampling,
    validate_statistics_profile,
)
from cnn_fpga.benchmark.phase9_powered_twin_runtime import HEARTBEAT_SCHEMA


SEAL_SCHEMA = "PHASE9-POWERED-TWIN-PREFORMAL-SEAL-V2"
VALIDATION_SCHEMA = "PHASE9-POWERED-TWIN-PREFORMAL-VALIDATION-V2"
RESOURCE_CONSUMPTION_SCHEMA = (
    "PHASE9-POWERED-TWIN-RESOURCE-CONSUMPTION-V1"
)
RESOURCE_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "run_id",
        "runner_id",
        "verdict",
        "config_sha256",
        "plan_sha256",
        "source_snapshot_sha256",
        "lineage_validation",
        "seed_firewall",
        "artifact_namespace",
        "formal_artifact_namespace_accessed",
        "full_size_receipt_count",
        "profile_measurements",
        "actual_peak_concurrency",
        "maximum_observed_worker_overlap",
        "resource_sample_count",
        "sample_interval_seconds",
        "sampling",
        "heartbeat",
        "streaming_statistics_dry_run",
        "joint_maxt_profile",
        "raw_seed_audit",
        "projection",
        "cell_projections",
        "maximum_inflight_temp_bytes",
        "analysis_scratch_bytes",
        "formal_projected_object_bytes",
        "formal_projected_wall_seconds",
        "inventory",
        "inventory_binding",
        "inventory_no_copy_evidence",
        "resource_gate_decision",
        "scientific_verdict",
        "qualified_claim",
        "claim_boundary",
        "attempt_witnesses_before_terminal",
        "analysis_sha256",
    }
)
SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "monotonic_seconds",
        "parent_pid",
        "parent_rss_bytes",
        "child_rss_bytes",
        "child_process_tree_pids",
        "live_child_count",
        "aggregate_rss_bytes",
        "stage",
        "previous_sample_sha256",
        "sample_sha256",
    }
)
SAMPLING_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "sample_count",
        "active_child_sample_count",
        "peak_aggregate_rss_bytes",
        "maximum_observed_live_children",
        "stage_peak_aggregate_rss_bytes",
        "first_sample",
        "last_sample",
        "peak_sample",
        "sample_chain_tip_sha256",
        "evidence",
        "summary_sha256",
    }
)
MEASUREMENT_FIELDS = frozenset(
    {
        "chunk_id",
        "plan_index",
        "pid",
        "wall_seconds",
        "object_bytes_unique",
        "explicit_alias_bytes",
        "conservative_payload_bytes",
        "object_bytes_by_role",
        "object_bindings",
        "expected_rows",
        "reset_rows",
        "receipt_sha256",
        "profile_peak_aggregate_rss_bytes",
    }
)
NO_COPY_FIELDS = frozenset(
    {
        "receipt_count",
        "unique_object_count",
        "object_bytes_unique",
        "object_tree_unchanged",
        "object_tree_sha256",
        "finalize_wall_seconds",
        "monolithic_archive",
        "merged_full_csv",
        "raw_payload_bytes_copied_during_finalize",
        "analysis_sha256",
    }
)
STATS_FIELDS = frozenset(
    {
        "schema_version", "gate_count", "replicates",
        "largest_cluster_count", "largest_density_dimension", "streaming",
        "maximum_coexisting_gate_buffers", "cached_cluster_root_groups",
        "cached_sign_bytes", "production_rademacher_generator_exercised",
        "conservative_dual_leg_max_exercised",
        "dual_leg_evaluation_count", "l1_accumulation_exercised",
        "largest_density_kernel_exercised", "largest_density_root_count",
        "largest_density_block_rows", "largest_density_block_count",
        "largest_density_source_buffer_count",
        "largest_density_rss_callback_count",
        "largest_density_perturbation_shape",
        "largest_density_perturbation_bytes",
        "largest_density_update_bytes",
        "largest_density_trace_norm_evaluations",
        "largest_density_kernel_sha256",
        "persistent_working_set_components",
        "persistent_working_set_bytes",
        "largest_density_peak_live_components",
        "largest_density_peak_live_bytes", "l1_maxima_sha256",
        "peak_explicit_working_set_bytes",
        "peak_analysis_scratch_bytes", "wall_seconds",
        "kernel_trace_sha256", "retained_density_physicality_profile",
        "seed_namespace", "seed_address",
        "formal_seed_addresses_accessed", "scientific_influences_used",
        "scientific_verdict", "qualified_claim",
        "profile_peak_aggregate_rss_bytes", "analysis_sha256",
    }
)
PHYSICALITY_FIELDS = frozenset(
    {
        "schema_version", "matrix_dimension", "block_size",
        "fixture_matrix_count", "fixture_bytes", "timed_repeats",
        "timed_matrix_evaluations", "trial_wall_seconds",
        "measured_total_wall_seconds", "trial_coefficient_of_variation",
        "worst_seconds_per_matrix", "projected_full_retained_count",
        "projected_full_serial_wall_seconds", "full_fixture_generated",
        "coverage_mode", "resource_profile_is_full_coverage",
        "formal_full_coverage_required",
        "complex64_to_complex128_exercised", "trace_recomputed",
        "hermiticity_frobenius_recomputed",
        "batched_minimum_eigvalsh_recomputed",
        "weyl_certificate_checked", "rss_callback_count",
        "peak_explicit_live_components", "peak_explicit_live_bytes",
        "kernel_sha256", "seed_namespace", "seed_address",
        "formal_seed_addresses_accessed", "scientific_data_used",
        "scientific_verdict", "qualified_claim", "analysis_sha256",
    }
)
VALIDATION_FIELDS = frozenset(
    {
        "schema_version", "source_snapshot_sha256",
        "resource_preflight", "resource_consumption_sha256",
        "attempt_ledger", "command", "python", "platform",
        "returncode", "elapsed_seconds", "stdout", "stderr",
        "stdout_sha256", "stderr_sha256", "verdict",
        "formal_outcomes_accessed", "claim_boundary",
        "analysis_sha256",
    }
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _binding(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    payload = resolved.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"conflicting immutable preformal artifact: {path}")
        return
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    f"conflicting preformal publication race: {path}"
                )
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    field: str = "analysis_sha256",
) -> None:
    claimed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if claimed != _sha(unsigned):
        raise RuntimeError(f"{field} mismatch")


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {label}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value


def _safe_run_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 96
        or any(
            character
            not in "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            for character in value
        )
    ):
        raise RuntimeError("unsafe resource run_id")
    return value


def _require_pass_resource_report(
    resource: Mapping[str, Any],
    *,
    config_sha256: str,
    plan_sha256: str,
    source_snapshot_sha256: str,
) -> str:
    """Reject incomplete/failed/minimal reports before any release work."""

    if set(resource) != RESOURCE_REPORT_FIELDS:
        raise RuntimeError("resource report exact top-level schema drift")
    _verify_self_hash(resource)
    run_id = _safe_run_id(resource.get("run_id"))
    claims = resource.get("claim_boundary")
    if (
        resource.get("schema_version") != PREFLIGHT_SCHEMA
        or resource.get("task_id") != TASK_ID
        or resource.get("runner_id") != RESOURCE_RUNNER_ID
        or resource.get("verdict") != PASS_VERDICT
        or resource.get("config_sha256") != config_sha256
        or resource.get("plan_sha256") != plan_sha256
        or resource.get("source_snapshot_sha256")
        != source_snapshot_sha256
        or resource.get("formal_artifact_namespace_accessed") is not False
        or resource.get("scientific_verdict") is not None
        or resource.get("qualified_claim") is not None
        or not isinstance(claims, Mapping)
        or set(claims) != set(EXPECTED_CLAIM_FIELDS)
        or any(value is not None for value in claims.values())
    ):
        raise RuntimeError("resource report identity/claim boundary drift")
    return run_id


def _verify_lineage_evidence(
    resource: Mapping[str, Any],
    *,
    config_sha256: str,
    plan_sha256: str,
    snapshot: Mapping[str, Any],
) -> None:
    lineage = resource.get("lineage_validation")
    expected_fields = {
        "config_sha256",
        "plan_sha256",
        "source_snapshot_sha256",
        "runtime_source_count",
        "validation_source_count",
        "claim_boundary_all_literal_null",
        "passed",
        "analysis_sha256",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != expected_fields:
        raise RuntimeError("resource lineage evidence schema drift")
    _verify_self_hash(lineage)
    if (
        lineage.get("config_sha256") != config_sha256
        or lineage.get("plan_sha256") != plan_sha256
        or lineage.get("source_snapshot_sha256")
        != snapshot["source_snapshot_sha256"]
        or lineage.get("runtime_source_count")
        != snapshot["runtime_source_count"]
        or lineage.get("validation_source_count")
        != snapshot["validation_source_count"]
        or lineage.get("claim_boundary_all_literal_null") is not True
        or lineage.get("passed") is not True
    ):
        raise RuntimeError("resource lineage evidence value drift")


def _verify_sampling_evidence(
    root: Path,
    preflight_root: Path,
    resource: Mapping[str, Any],
    *,
    heartbeat_period_seconds: float,
) -> tuple[
    dict[str, Any],
    set[int],
    set[int],
    dict[str, float],
]:
    summary = resource.get("sampling")
    if not isinstance(summary, Mapping) or set(summary) != SAMPLING_SUMMARY_FIELDS:
        raise RuntimeError("resource sampling summary schema drift")
    _verify_self_hash(summary, field="summary_sha256")
    evidence_path = preflight_root / "resource_samples.jsonl"
    if summary.get("evidence") != _binding(evidence_path, root):
        raise RuntimeError("resource sampling live binding drift")
    raw_lines = evidence_path.read_bytes().splitlines(keepends=True)
    if not raw_lines or any(not line.endswith(b"\n") for line in raw_lines):
        raise RuntimeError("resource sampling JSONL framing drift")
    records: list[dict[str, Any]] = []
    previous = "0" * 64
    parent_pid: int | None = None
    active_count = 0
    peak_rss = 0
    peak_children = 0
    peak_record: dict[str, Any] | None = None
    stage_peaks: dict[str, int] = {}
    prior_time: float | None = None
    maximum_gap = 0.0
    concurrent_pids: set[int] = set()
    representative_pids: set[int] = set()
    stage_first_times: dict[str, float] = {}
    allowed_stages = [
        "starting",
        "formal_lpt_four_worker_peak",
        "representative_four_worker_profiles",
        "joint_maxt_3037x199",
        "inventory_finalize_no_copy",
    ]
    stage_rank = {name: index for index, name in enumerate(allowed_stages)}
    previous_stage_rank = 0
    for sequence, raw_line in enumerate(raw_lines):
        record = _strict_json_bytes(
            raw_line,
            label=f"resource sample line {sequence + 1}",
        )
        if set(record) != SAMPLE_FIELDS:
            raise RuntimeError("resource sample exact schema drift")
        claimed = record.get("sample_sha256")
        unsigned = dict(record)
        unsigned.pop("sample_sha256", None)
        child_rss = record.get("child_rss_bytes")
        child_trees = record.get("child_process_tree_pids")
        stage = record.get("stage")
        monotonic = record.get("monotonic_seconds")
        if (
            record.get("schema_version") != SAMPLING_SCHEMA
            or record.get("sequence") != sequence
            or record.get("previous_sample_sha256") != previous
            or claimed != _sha(unsigned)
            or isinstance(record.get("parent_pid"), bool)
            or not isinstance(record.get("parent_pid"), int)
            or int(record["parent_pid"]) <= 0
            or isinstance(record.get("parent_rss_bytes"), bool)
            or not isinstance(record.get("parent_rss_bytes"), int)
            or int(record["parent_rss_bytes"]) <= 0
            or not isinstance(child_rss, Mapping)
            or not isinstance(child_trees, Mapping)
            or set(child_rss) != set(child_trees)
            or stage not in stage_rank
            or isinstance(monotonic, bool)
            or not isinstance(monotonic, (int, float))
            or not math.isfinite(float(monotonic))
        ):
            raise RuntimeError("resource sample lineage/value drift")
        if parent_pid is None:
            parent_pid = int(record["parent_pid"])
        elif int(record["parent_pid"]) != parent_pid:
            raise RuntimeError("resource sampling parent PID drift")
        child_keys: set[int] = set()
        tree_members: set[int] = set()
        for key, rss in child_rss.items():
            try:
                pid = int(key)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("resource child PID key drift") from exc
            if str(pid) != key or pid <= 0:
                raise RuntimeError("resource child PID is not canonical")
            if isinstance(rss, bool) or not isinstance(rss, int) or rss <= 0:
                raise RuntimeError("resource child RSS drift")
            members = child_trees[key]
            if (
                not isinstance(members, list)
                or not members
                or any(
                    isinstance(member, bool)
                    or not isinstance(member, int)
                    or member <= 0
                    for member in members
                )
                or len(members) != len(set(members))
                or pid not in members
                or tree_members.intersection(members)
            ):
                raise RuntimeError("resource child process tree drift")
            child_keys.add(pid)
            tree_members.update(members)
        aggregate = int(record["parent_rss_bytes"]) + sum(
            int(value) for value in child_rss.values()
        )
        if (
            record.get("live_child_count") != len(child_rss)
            or record.get("aggregate_rss_bytes") != aggregate
        ):
            raise RuntimeError("resource sample RSS arithmetic drift")
        current_time = float(monotonic)
        stage_first_times.setdefault(str(stage), current_time)
        if prior_time is not None:
            if current_time <= prior_time:
                raise RuntimeError("resource sampling time is not increasing")
            maximum_gap = max(maximum_gap, current_time - prior_time)
        prior_time = current_time
        current_rank = stage_rank[str(stage)]
        if current_rank < previous_stage_rank:
            raise RuntimeError("resource sampling stage regressed")
        previous_stage_rank = current_rank
        if child_rss:
            active_count += 1
        if stage == "formal_lpt_four_worker_peak":
            concurrent_pids.update(child_keys)
        elif stage == "representative_four_worker_profiles":
            representative_pids.update(child_keys)
        peak_children = max(peak_children, len(child_rss))
        stage_peaks[str(stage)] = max(
            stage_peaks.get(str(stage), 0), aggregate
        )
        if aggregate >= peak_rss:
            peak_rss = aggregate
            peak_record = dict(record)
        previous = str(claimed)
        records.append(record)
    interval = float(resource.get("sample_interval_seconds", math.nan))
    if (
        interval != 5.0
        or maximum_gap > max(4.0 * interval, heartbeat_period_seconds)
        or set(stage_peaks) != set(allowed_stages)
    ):
        raise RuntimeError("resource sampling cadence/stage coverage drift")
    recomputed: dict[str, Any] = {
        "schema_version": SAMPLING_SCHEMA,
        "sample_count": len(records),
        "active_child_sample_count": active_count,
        "peak_aggregate_rss_bytes": peak_rss,
        "maximum_observed_live_children": peak_children,
        "stage_peak_aggregate_rss_bytes": stage_peaks,
        "first_sample": records[0],
        "last_sample": records[-1],
        "peak_sample": peak_record,
        "sample_chain_tip_sha256": previous,
        "evidence": _binding(evidence_path, root),
    }
    recomputed["summary_sha256"] = _sha(recomputed)
    if dict(summary) != recomputed:
        raise RuntimeError("resource sampling summary/raw chain drift")
    validate_continuous_sampling(recomputed)
    if (
        resource.get("resource_sample_count") != len(records)
        or resource.get("actual_peak_concurrency") != 4
        or resource.get("maximum_observed_worker_overlap") != 4
        or peak_children != 4
        or len(concurrent_pids) != 4
        or len(representative_pids) != 4
        or bool(concurrent_pids & representative_pids)
    ):
        raise RuntimeError("resource sampling top-level concurrency drift")
    stage_windows: dict[str, float] = {}
    for index, stage in enumerate(allowed_stages[:-1]):
        next_stage = allowed_stages[index + 1]
        duration = stage_first_times[next_stage] - stage_first_times[stage]
        if not math.isfinite(duration) or duration <= 0.0:
            raise RuntimeError("resource sampling stage window is invalid")
        stage_windows[stage] = duration
    return recomputed, concurrent_pids, representative_pids, stage_windows


def _verify_heartbeat_evidence(
    root: Path,
    preflight_root: Path,
    resource: Mapping[str, Any],
    sampling: Mapping[str, Any],
    *,
    expected_period_seconds: float,
) -> dict[str, Any]:
    wrapper = resource.get("heartbeat")
    wrapper_fields = {
        "path",
        "binding",
        "period_seconds",
        "latest_sequence",
        "observed_sampling_span_seconds",
        "latest_child_pids",
        "independent_of_chunk_completion",
    }
    if not isinstance(wrapper, Mapping) or set(wrapper) != wrapper_fields:
        raise RuntimeError("resource heartbeat wrapper schema drift")
    path = preflight_root / "heartbeat.json"
    expected_path = path.resolve().relative_to(root.resolve()).as_posix()
    heartbeat = _strict_json(path)
    _verify_self_hash(heartbeat, field="heartbeat_sha256")
    if set(heartbeat) != {
        "schema_version",
        "run_id",
        "owner_token",
        "pid",
        "process_creation_time",
        "sequence",
        "monotonic_seconds",
        "wall_time_ns",
        "snapshot",
        "heartbeat_sha256",
    }:
        raise RuntimeError("resource heartbeat live schema drift")
    snapshot = heartbeat.get("snapshot")
    span = (
        float(sampling["last_sample"]["monotonic_seconds"])
        - float(sampling["first_sample"]["monotonic_seconds"])
    )
    parent_pid = int(sampling["first_sample"]["parent_pid"])
    owner_token = heartbeat.get("owner_token")
    creation_time = heartbeat.get("process_creation_time")
    sequence = heartbeat.get("sequence")
    monotonic = heartbeat.get("monotonic_seconds")
    wall_time_ns = heartbeat.get("wall_time_ns")
    if (
        heartbeat.get("schema_version") != HEARTBEAT_SCHEMA
        or heartbeat.get("run_id") != resource["run_id"]
        or heartbeat.get("pid") != parent_pid
        or not isinstance(owner_token, str)
        or len(owner_token) != 32
        or any(character not in "0123456789abcdef" for character in owner_token)
        or isinstance(creation_time, bool)
        or not isinstance(creation_time, (int, float))
        or not math.isfinite(float(creation_time))
        or float(creation_time) <= 0.0
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
        or isinstance(monotonic, bool)
        or not isinstance(monotonic, (int, float))
        or not math.isfinite(float(monotonic))
        or float(monotonic)
        < float(sampling["last_sample"]["monotonic_seconds"])
        or isinstance(wall_time_ns, bool)
        or not isinstance(wall_time_ns, int)
        or wall_time_ns <= 0
        or not isinstance(snapshot, Mapping)
        or set(snapshot) != {"stage", "child_pids", "profiles_completed"}
        or snapshot.get("stage") != "inventory_finalize_no_copy"
        or snapshot.get("child_pids") != []
        or snapshot.get("profiles_completed") != 8
        or wrapper.get("path") != expected_path
        or wrapper.get("binding") != _binding(path, root)
        or wrapper.get("period_seconds") != expected_period_seconds
        or wrapper.get("latest_sequence") != heartbeat["sequence"]
        or wrapper.get("latest_child_pids") != []
        or wrapper.get("independent_of_chunk_completion") is not True
        or not math.isclose(
            float(wrapper.get("observed_sampling_span_seconds", math.nan)),
            span,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or span < float(wrapper["period_seconds"])
    ):
        raise RuntimeError("resource heartbeat/live sampling binding drift")
    return heartbeat


def _measurement_from_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    objects = receipt["objects"]
    unique = {
        str(binding["sha256"]): int(binding["bytes"])
        for binding in objects
    }
    by_role = {str(binding["role"]): binding for binding in objects}
    alias = 0
    primary = by_role.get("primary_density_npy")
    expected = by_role.get("rb_expected_density_npy")
    if (
        receipt["cell"].get("layer") in {"shared", "probe"}
        and primary is not None
        and expected is not None
        and primary["path"] == expected["path"]
        and primary["sha256"] == expected["sha256"]
        and primary["bytes"] == expected["bytes"]
    ):
        alias = int(expected["bytes"])
    return {
        "chunk_id": receipt["cell"]["chunk_id"],
        "plan_index": int(receipt["cell"]["plan_index"]),
        "object_bytes_unique": sum(unique.values()),
        "explicit_alias_bytes": alias,
        "conservative_payload_bytes": (
            sum(int(binding["bytes"]) for binding in objects) - alias
        ),
        "object_bytes_by_role": {
            str(binding["role"]): int(binding["bytes"])
            for binding in objects
        },
        "object_bindings": [
            {
                "role": str(binding["role"]),
                "bytes": int(binding["bytes"]),
                "sha256": str(binding["sha256"]),
            }
            for binding in objects
        ],
        "expected_rows": int(receipt["diagnostics"]["expected_rows"]),
        "reset_rows": int(receipt["diagnostics"]["reset_rows"]),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _verify_measurements(
    resource: Mapping[str, Any],
    receipts: Mapping[int, Mapping[str, Any]],
    sampling: Mapping[str, Any],
    *,
    concurrent_pids: set[int],
    representative_pids: set[int],
    stage_windows: Mapping[str, float],
) -> list[dict[str, Any]]:
    measurements = resource.get("profile_measurements")
    if (
        not isinstance(measurements, list)
        or len(measurements) != 8
        or any(
            not isinstance(item, Mapping)
            or set(item) != MEASUREMENT_FIELDS
            for item in measurements
        )
        or [item["plan_index"] for item in measurements]
        != [388, 389, 403, 478, 480, 482, 484, 507]
    ):
        raise RuntimeError("resource measurement exact coverage drift")
    stage_peaks = sampling["stage_peak_aggregate_rss_bytes"]
    stage_measurement_walls: dict[str, list[float]] = {
        "formal_lpt_four_worker_peak": [],
        "representative_four_worker_profiles": [],
    }
    stage_measurement_pids: dict[str, set[int]] = {
        "formal_lpt_four_worker_peak": set(),
        "representative_four_worker_profiles": set(),
    }
    for item in measurements:
        index = int(item["plan_index"])
        expected = _measurement_from_receipt(receipts[index])
        for key, value in expected.items():
            if item.get(key) != value:
                raise RuntimeError(
                    f"resource measurement/live receipt drift at {index}:{key}"
                )
        pid = item.get("pid")
        wall = item.get("wall_seconds")
        peak = item.get("profile_peak_aggregate_rss_bytes")
        peak_indices = {478, 480, 482, 484}
        stage_name = (
            "formal_lpt_four_worker_peak"
            if index in peak_indices
            else "representative_four_worker_profiles"
        )
        expected_peak = stage_peaks[stage_name]
        expected_pid_set = (
            concurrent_pids
            if index in peak_indices
            else representative_pids
        )
        stage_window = float(stage_windows[stage_name])
        if isinstance(wall, (int, float)) and not isinstance(wall, bool):
            stage_measurement_walls[stage_name].append(float(wall))
        if isinstance(pid, int) and not isinstance(pid, bool):
            stage_measurement_pids[stage_name].add(pid)
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or pid not in expected_pid_set
            or isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) <= 0.0
            or float(wall) > stage_window
            or isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak != expected_peak
        ):
            raise RuntimeError("resource measurement runtime witness drift")
    for stage_name, walls in stage_measurement_walls.items():
        longest = max(walls, default=0.0)
        stage_window = float(stage_windows[stage_name])
        if (
            longest <= 0.0
            or stage_window
            > longest + max(30.0, 0.20 * longest)
        ):
            raise RuntimeError("resource measurement/stage wall binding drift")
    if (
        stage_measurement_pids["formal_lpt_four_worker_peak"]
        != concurrent_pids
        or stage_measurement_pids["representative_four_worker_profiles"]
        != representative_pids
    ):
        raise RuntimeError("resource measurement/stage PID bijection drift")
    return [dict(item) for item in measurements]


def _verify_attempt_chain(
    root: Path,
    preflight_root: Path,
    resource: Mapping[str, Any],
    heartbeat: Mapping[str, Any],
) -> dict[str, Any]:
    path = preflight_root / "attempts.jsonl"
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != 2 or any(
        not line.endswith(b"\n") for line in raw_lines
    ):
        raise RuntimeError("resource attempt ledger is not exact START/PASS")
    records: list[dict[str, Any]] = []
    previous = "0" * 64
    for sequence, line in enumerate(raw_lines):
        record = _strict_json_bytes(
            line, label=f"resource attempt line {sequence + 1}"
        )
        if set(record) != {
            "schema_version",
            "task_id",
            "run_id",
            "sequence",
            "previous_event_sha256",
            "event",
            "payload",
            "event_sha256",
        }:
            raise RuntimeError("resource attempt exact schema drift")
        claimed = record.get("event_sha256")
        unsigned = dict(record)
        unsigned.pop("event_sha256", None)
        if (
            record.get("schema_version") != ATTEMPT_SCHEMA
            or record.get("task_id") != TASK_ID
            or record.get("run_id") != resource["run_id"]
            or record.get("sequence") != sequence
            or record.get("previous_event_sha256") != previous
            or claimed != _sha(unsigned)
        ):
            raise RuntimeError("resource attempt hash/lineage drift")
        witness = (
            preflight_root
            / "attempt_events"
            / f"{sequence:08d}.json"
        )
        if witness.read_bytes() != line:
            raise RuntimeError("resource attempt immutable witness drift")
        records.append(record)
        previous = str(claimed)
    if (
        records[0]["event"] != "START_RESOURCE_PREFLIGHT"
        or records[0]["payload"]
        != {
            "formal_seed_addresses_accessed": False,
            "artifact_namespace": resource["artifact_namespace"],
            "owner_token": heartbeat["owner_token"],
            "owner_pid": heartbeat["pid"],
            "process_creation_time": heartbeat[
                "process_creation_time"
            ],
        }
        or records[1]["event"] != "PASS_RESOURCE_PREFLIGHT"
        or records[1]["payload"]
        != {
            "analysis_sha256": resource["analysis_sha256"],
            "formal_seed_addresses_accessed": False,
        }
        or resource.get("attempt_witnesses_before_terminal")
        != [
            _binding(
                preflight_root / "attempt_events" / "00000000.json",
                root,
            )
        ]
        or len(
            list((preflight_root / "attempt_events").glob("*.json"))
        )
        != 2
    ):
        raise RuntimeError("resource attempt START/PASS semantic drift")
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "binding": _binding(path, root),
        "event_count": 2,
        "chain_tip_sha256": previous,
        "start_witness": _binding(
            preflight_root / "attempt_events" / "00000000.json", root
        ),
        "pass_witness": _binding(
            preflight_root / "attempt_events" / "00000001.json", root
        ),
    }


def validate_resource_release_evidence(
    root: Path,
    config: Mapping[str, Any],
    config_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    resource_path: Path,
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently consume every live resource artifact before release."""

    root = root.resolve()
    plan_sha = str(plan["canonical_plan_sha256"])
    run_id = _require_pass_resource_report(
        resource,
        config_sha256=str(config_binding["sha256"]),
        plan_sha256=plan_sha,
        source_snapshot_sha256=str(snapshot["source_snapshot_sha256"]),
    )
    _verify_lineage_evidence(
        resource,
        config_sha256=str(config_binding["sha256"]),
        plan_sha256=plan_sha,
        snapshot=snapshot,
    )
    preflight_root, expected_namespace = isolated_preflight_paths(
        root, config, run_id=run_id
    )
    if resource.get("artifact_namespace") != expected_namespace:
        raise RuntimeError("resource isolated artifact namespace drift")
    if (preflight_root / "resource_preflight_failed.json").exists():
        raise RuntimeError("resource namespace contains a failure terminal")
    if (preflight_root / "owner.lock").exists():
        raise RuntimeError("resource supervisor is still active")
    expected_firewall = assert_seed_firewall(config)
    if resource.get("seed_firewall") != expected_firewall:
        raise RuntimeError("resource seed firewall/report drift")
    cells = build_cell_plan(config)
    formal_peak, representatives = profile_cells(config, cells)
    selected = formal_peak + representatives
    transaction_directories = [
        root / expected_namespace["object_store"],
        root / expected_namespace["staging_directory"],
        root / expected_namespace["receipt_directory"],
    ]
    if any(
        not path.exists()
        or not path.is_dir()
        or path.is_symlink()
        for path in transaction_directories
    ):
        raise RuntimeError(
            "resource transaction directories are not live regular directories"
        )
    store = ImmutableObjectStore(
        repository_root=root,
        object_root=root / expected_namespace["object_store"],
        staging_root=root / expected_namespace["staging_directory"],
        receipt_root=root / expected_namespace["receipt_directory"],
        task_id=TASK_ID,
        run_id=run_id,
        config_sha256=str(config_binding["sha256"]),
        plan_sha256=plan_sha,
        source_snapshot_sha256=str(snapshot["source_snapshot_sha256"]),
        seed_namespace="resource_preflight",
        runner_id=RAW_RUNNER_ID,
    )
    raw_seed_audit = audit_resource_profile_receipts(
        root, config, store, selected
    )
    if resource.get("raw_seed_audit") != raw_seed_audit:
        raise RuntimeError("resource raw seed audit/live evidence drift")
    live_inventory = store.inventory([asdict(cell) for cell in selected])
    inventory_path = preflight_root / "inventory.json"
    file_inventory = _strict_json(inventory_path)
    if (
        resource.get("inventory") != live_inventory
        or file_inventory != live_inventory
        or resource.get("inventory_binding")
        != _binding(inventory_path, root)
        or live_inventory.get("schema_version") != INVENTORY_SCHEMA
        or live_inventory.get("receipt_count") != 8
        or live_inventory.get("totals")
        != {
            "expected_rows": 227_328,
            "observed_rows": 227_328,
            "exception_rows": 0,
            "missing_rows": 0,
            "conservation_failures": 0,
            "reset_rows": 15_360,
            "reset_sidecar_rows": 15_360,
            "object_bytes_unique": live_inventory["totals"][
                "object_bytes_unique"
            ],
        }
        or live_inventory.get("raw_status")
        != "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT"
        or resource.get("full_size_receipt_count")
        != live_inventory["receipt_count"]
    ):
        raise RuntimeError("resource inventory/live receipt closure drift")
    receipt_by_index: dict[int, Mapping[str, Any]] = {}
    for cell in selected:
        receipt_by_index[cell.plan_index] = store.verify_receipt(
            store.receipt_path(cell.chunk_id),
            expected_cell=asdict(cell),
        )
    heartbeat_period = float(
        config["runtime_contract"]["heartbeat_period_seconds"]
    )
    (
        sampling,
        concurrent_pids,
        representative_pids,
        stage_windows,
    ) = _verify_sampling_evidence(
        root,
        preflight_root,
        resource,
        heartbeat_period_seconds=heartbeat_period,
    )
    heartbeat = _verify_heartbeat_evidence(
        root,
        preflight_root,
        resource,
        sampling,
        expected_period_seconds=heartbeat_period,
    )
    measurements = _verify_measurements(
        resource,
        receipt_by_index,
        sampling,
        concurrent_pids=concurrent_pids,
        representative_pids=representative_pids,
        stage_windows=stage_windows,
    )
    stats = resource.get("streaming_statistics_dry_run")
    if (
        not isinstance(stats, Mapping)
        or set(stats) != STATS_FIELDS
        or resource.get("joint_maxt_profile") != stats
    ):
        raise RuntimeError("resource statistics duplicate view drift")
    _verify_self_hash(stats)
    physicality = stats.get("retained_density_physicality_profile")
    if (
        not isinstance(physicality, Mapping)
        or set(physicality) != PHYSICALITY_FIELDS
    ):
        raise RuntimeError("resource retained-density profile missing")
    _verify_self_hash(physicality)
    validate_statistics_profile(config, stats)
    measured_statistics_stage_wall = (
        float(stats["wall_seconds"])
        + float(physicality["measured_total_wall_seconds"])
    )
    sampled_statistics_stage_wall = float(
        stage_windows["joint_maxt_3037x199"]
    )
    if (
        stats.get("profile_peak_aggregate_rss_bytes")
        != sampling["stage_peak_aggregate_rss_bytes"][
            "joint_maxt_3037x199"
        ]
        or resource.get("analysis_scratch_bytes")
        != stats["peak_analysis_scratch_bytes"]
        or measured_statistics_stage_wall > sampled_statistics_stage_wall
        or sampled_statistics_stage_wall
        > measured_statistics_stage_wall
        + max(30.0, 0.20 * measured_statistics_stage_wall)
    ):
        raise RuntimeError("resource statistics/RSS scratch binding drift")
    projection = stratified_projection(
        config,
        cells,
        measurements,
        stats_wall_seconds=float(stats["wall_seconds"]),
        retained_density_physicality_wall_seconds=float(
            physicality["projected_full_serial_wall_seconds"]
        ),
        inventory_finalize_wall_seconds=float(
            resource["inventory_no_copy_evidence"][
                "finalize_wall_seconds"
            ]
        ),
        inventory_profile_object_bytes=int(
            live_inventory["totals"]["object_bytes_unique"]
        ),
        inventory_profile_receipt_count=int(
            live_inventory["receipt_count"]
        ),
    )
    if (
        resource.get("projection") != projection
        or resource.get("cell_projections")
        != projection["cell_projections"]
        or resource.get("formal_projected_object_bytes")
        != projection["projected_formal_artifact_bytes"]
        or resource.get("formal_projected_wall_seconds")
        != projection[
            "projected_formal_wall_seconds_at_frozen_concurrency"
        ]
    ):
        raise RuntimeError("resource projection recomputation drift")
    maximum_inflight = sum(
        sorted(
            (
                int(item["projected_transient_bytes"])
                for item in projection["cell_projections"]
            ),
            reverse=True,
        )[: int(config["runtime_contract"]["max_workers"])]
    )
    scratch = int(stats["peak_analysis_scratch_bytes"])
    if (
        resource.get("maximum_inflight_temp_bytes") != maximum_inflight
        or resource.get("analysis_scratch_bytes") != scratch
    ):
        raise RuntimeError("resource inflight/scratch byte accounting drift")
    decision = resource.get("resource_gate_decision")
    if not isinstance(decision, Mapping):
        raise RuntimeError("resource gate decision missing")
    _verify_self_hash(decision, field="decision_sha256")
    limits = {
        "maximum_peak_rss_bytes": int(
            config["resource_contract"]["maximum_peak_rss_bytes"]
        ),
        "maximum_artifact_bytes": int(
            config["resource_contract"]["maximum_artifact_bytes"]
        ),
        "minimum_post_projection_free_bytes": int(
            config["resource_contract"][
                "minimum_post_projection_free_bytes"
            ]
        ),
        "maximum_wall_seconds": float(
            config["resource_contract"]["maximum_wall_seconds"]
        ),
    }
    stored_free = decision.get("disk_free_bytes")
    if isinstance(stored_free, bool) or not isinstance(stored_free, int):
        raise RuntimeError("resource stored disk-free witness drift")
    expected_post_free = (
        stored_free
        - int(projection["projected_formal_artifact_bytes"])
        - maximum_inflight
        - scratch
    )
    expected_checks = {
        "rss": int(sampling["peak_aggregate_rss_bytes"])
        <= limits["maximum_peak_rss_bytes"],
        "artifact": int(projection["projected_formal_artifact_bytes"])
        <= limits["maximum_artifact_bytes"],
        "disk": expected_post_free
        >= limits["minimum_post_projection_free_bytes"],
        "wall": float(
            projection[
                "projected_formal_wall_seconds_at_frozen_concurrency"
            ]
        )
        <= limits["maximum_wall_seconds"],
        "inventory": True,
    }
    expected_decision = {
        "checks": expected_checks,
        "passed": all(expected_checks.values()),
        "disk_free_bytes": stored_free,
        "projected_post_formal_free_bytes": expected_post_free,
        "maximum_inflight_temp_bytes": maximum_inflight,
        "analysis_scratch_bytes": scratch,
        "limits": limits,
    }
    expected_decision["decision_sha256"] = _sha(expected_decision)
    if dict(decision) != expected_decision or decision["passed"] is not True:
        raise RuntimeError("resource gate decision recomputation drift")
    live_decision = resource_gate_decision(
        config,
        sampling=sampling,
        projection=projection,
        inventory=live_inventory,
        run_directory=preflight_root,
        maximum_inflight_temp_bytes=maximum_inflight,
        analysis_scratch_bytes=scratch,
    )
    if live_decision["passed"] is not True:
        raise RuntimeError("resource live admission no longer passes")
    no_copy = resource.get("inventory_no_copy_evidence")
    if (
        not isinstance(no_copy, Mapping)
        or set(no_copy) != NO_COPY_FIELDS
    ):
        raise RuntimeError("resource no-copy evidence missing")
    _verify_self_hash(no_copy)
    tree = _object_tree_snapshot(store.object_root)
    if (
        no_copy.get("receipt_count") != live_inventory["receipt_count"]
        or no_copy.get("unique_object_count")
        != live_inventory["unique_object_count"]
        or no_copy.get("object_bytes_unique")
        != live_inventory["totals"]["object_bytes_unique"]
        or no_copy.get("object_tree_unchanged") is not True
        or no_copy.get("object_tree_sha256") != _sha(tree)
        or isinstance(no_copy.get("finalize_wall_seconds"), bool)
        or not isinstance(
            no_copy.get("finalize_wall_seconds"), (int, float)
        )
        or not math.isfinite(float(no_copy["finalize_wall_seconds"]))
        or float(no_copy["finalize_wall_seconds"]) < 0.0
        or no_copy.get("monolithic_archive") is not None
        or no_copy.get("merged_full_csv") is not None
        or no_copy.get("raw_payload_bytes_copied_during_finalize") != 0
    ):
        raise RuntimeError("resource no-copy/live object tree drift")
    forbidden_names = {"merged.csv", "full.csv", "all_rows.csv"}
    for transaction_root in (
        store.object_root,
        store.receipt_root,
        store.staging_root,
    ):
        if any(
            path.is_file()
            and (
                path.suffix.lower() == ".zip"
                or path.name.lower() in forbidden_names
            )
            for path in transaction_root.rglob("*")
        ):
            raise RuntimeError("resource monolithic/merged artifact found")
    attempts = _verify_attempt_chain(
        root, preflight_root, resource, heartbeat
    )
    evidence: dict[str, Any] = {
        "schema_version": RESOURCE_CONSUMPTION_SCHEMA,
        "resource_report": _binding(resource_path, root),
        "run_id": run_id,
        "config_sha256": str(config_binding["sha256"]),
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": str(
            snapshot["source_snapshot_sha256"]
        ),
        "receipt_count": 8,
        "ledger_rows_verified": 227_328,
        "reset_rows_verified": 15_360,
        "formal_seed_addresses_accessed": False,
        "live_object_count": live_inventory["unique_object_count"],
        "live_object_bytes": live_inventory["totals"][
            "object_bytes_unique"
        ],
        "sampling_records_verified": sampling["sample_count"],
        "maximum_observed_worker_overlap": 4,
        "heartbeat_sequence": heartbeat["sequence"],
        "projection_cells_verified": 518,
        "joint_maxt_gate_count": 3037,
        "joint_maxt_replicates": 199,
        "attempt_chain": attempts,
        "live_resource_admission_passed": True,
        "scientific_verdict": None,
        "qualified_claim": None,
        "claim_boundary": {
            field: None for field in EXPECTED_CLAIM_FIELDS
        },
    }
    evidence["analysis_sha256"] = _sha(evidence)
    return evidence


def _assert_no_formal_outcome(root: Path, config: Mapping[str, Any]) -> None:
    paths = config["artifact_paths"]
    object_root = root / str(paths["object_store"])
    receipt_root = root / str(paths["receipt_directory"])
    staging_root = root / str(paths["staging_directory"])
    if object_root.exists() and (
        not object_root.is_dir()
        or any(path.is_file() for path in object_root.rglob("*"))
    ):
        raise RuntimeError("formal content object exists before preformal seal")
    if receipt_root.exists() and (
        not receipt_root.is_dir() or any(receipt_root.glob("*.json"))
    ):
        raise RuntimeError("formal receipt exists before preformal seal")
    if staging_root.exists() and (
        not staging_root.is_dir()
        or any(path.is_file() for path in staging_root.rglob("*"))
    ):
        raise RuntimeError("formal staging payload exists before preformal seal")
    for name in ("inventory", "execution_manifest", "independent_verification"):
        if (root / str(paths[name])).exists():
            raise RuntimeError(f"formal {name} exists before preformal seal")


def run_focused_validation(
    root: Path,
    config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    resource_binding: Mapping[str, Any],
    resource_consumption: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all byte-bound T04 tests in the frozen numerical environment."""

    paths = config["artifact_paths"]
    report_path = root / str(paths["preformal_validation"])
    if report_path.exists():
        report = _strict_json(report_path)
        _verify_self_hash(report)
        if (
            set(report) != VALIDATION_FIELDS
            or report.get("schema_version") != VALIDATION_SCHEMA
            or report.get("source_snapshot_sha256")
            != snapshot["source_snapshot_sha256"]
            or report.get("resource_preflight") != resource_binding
            or report.get("resource_consumption_sha256")
            != resource_consumption["analysis_sha256"]
            or report.get("attempt_ledger")
            != resource_consumption["attempt_chain"]["binding"]
            or report.get("returncode") != 0
            or report.get("verdict") != "PASS_FOCUSED_ANTISIMPLIFICATION"
            or report.get("formal_outcomes_accessed") is not False
            or not isinstance(report.get("claim_boundary"), Mapping)
            or set(report["claim_boundary"])
            != set(EXPECTED_CLAIM_FIELDS)
            or any(
                report["claim_boundary"][field] is not None
                for field in EXPECTED_CLAIM_FIELDS
            )
            or report.get("python") != list(sys.version_info[:3])
            or report.get("platform") != platform.platform()
            or report.get("stdout_sha256")
            != sha256(str(report.get("stdout", "")).encode("utf-8")).hexdigest()
            or report.get("stderr_sha256")
            != sha256(str(report.get("stderr", "")).encode("utf-8")).hexdigest()
        ):
            raise RuntimeError("existing preformal validation is not reusable")
        return report
    run_directory = (root / str(paths["run_directory"])).resolve()
    run_directory.relative_to(root.resolve())
    base_temp = run_directory / "preformal_pytest_tmp"
    if base_temp.exists():
        raise RuntimeError("preformal pytest temp already exists; archive first")
    validation_paths = [
        str(root / str(value))
        for value in config["runtime_sources"]["validation_paths"]
    ]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *validation_paths,
        "-q",
        f"--basetemp={base_temp}",
    ]
    environment = dict(os.environ)
    expected_threads = config["runtime_contract"]["preimport_thread_environment"]
    if any(environment.get(key) != value for key, value in expected_threads.items()):
        raise RuntimeError("preformal validation thread environment drifted")
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    if base_temp.exists():
        resolved = base_temp.resolve()
        resolved.relative_to(run_directory)
        shutil.rmtree(resolved)
    report: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "resource_preflight": dict(resource_binding),
        "resource_consumption_sha256": resource_consumption[
            "analysis_sha256"
        ],
        "attempt_ledger": resource_consumption["attempt_chain"]["binding"],
        "command": command,
        "python": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "returncode": int(completed.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "verdict": (
            "PASS_FOCUSED_ANTISIMPLIFICATION"
            if completed.returncode == 0
            else "FAIL_FOCUSED_ANTISIMPLIFICATION"
        ),
        "formal_outcomes_accessed": False,
        "claim_boundary": {
            field: None for field in EXPECTED_CLAIM_FIELDS
        },
    }
    report["analysis_sha256"] = _sha(report)
    _immutable_json(report_path, report)
    if completed.returncode != 0:
        raise RuntimeError("focused preformal validation failed")
    return report


def create_preformal_seal(root: Path) -> dict[str, Any]:
    config, config_binding = load_config(root)
    _assert_no_formal_outcome(root, config)
    plan = plan_payload(config)
    plan_sha = str(plan["canonical_plan_sha256"])
    seed_registry = seed_registry_payload(config)
    snapshot = runtime_source_snapshot(root, config)
    paths = config["artifact_paths"]
    historical_scan = historical_seed_scan(
        root, root / str(config_binding["path"])
    )
    live_plan = _strict_json(root / str(paths["plan"]))
    live_seed_registry = _strict_json(root / str(paths["seed_registry"]))
    live_historical_scan = _strict_json(
        root / str(paths["historical_seed_scan"])
    )
    contract_path = root / str(paths["contract_preflight"])
    contract = _strict_json(contract_path)
    _verify_self_hash(contract)
    contract_claims = contract.get("claim_boundary")
    contract_gates = contract.get("gates")
    contract_bindings = contract.get("bindings")
    expected_contract_gates = {
        "C01_all_parent_bytes_verified",
        "C02_t03_t05_t06_semantic_chain_verified",
        "C03_exact_518_cell_plan",
        "C04_exact_2085888_row_denominator",
        "C05_exact_482304_primary_densities",
        "C06_fault_state_major_6x768",
        "C07_historical_seed_scan_recomputed",
        "C08_actual_seed_addresses_injective",
        "C09_content_addressed_no_zip_contract_frozen",
        "C10_all_claims_null_and_scientific_execution_blocked",
        "C11_all_runtime_and_validation_sources_live_regular",
    }
    expected_parent_checks = {
        "P01_t03_design_repair_independent_pass",
        "P02_t05_statistical_no_go_preserved",
        "P03_t06_independent_count_pass",
        "P04_selected_count_exact",
        "P05_selected_blueprint_exact",
        "P06_blueprint_counts_consume_selected_count",
        "P07_parent_claims_remain_null",
    }
    parent_checks = contract.get("parent_semantic_checks")
    expected_contract_bindings = {
        "config": config_binding,
        "plan": _binding(root / str(paths["plan"]), root),
        "seed_registry": _binding(
            root / str(paths["seed_registry"]), root
        ),
        "historical_seed_scan": _binding(
            root / str(paths["historical_seed_scan"]), root
        ),
    }
    if (
        contract.get("schema_version")
        != "PHASE9-POWERED-TWIN-CONTRACT-PREFLIGHT-V5"
        or contract.get("task_id") != TASK_ID
        or contract.get("status") != "PASS_OUTCOME_FREE_CONTRACT_PREFLIGHT"
        or contract.get("plan_summary", {}).get("plan_sha256") != plan_sha
        or contract.get("formal_outcomes_accessed") is not False
        or contract.get("scientific_execution_released") is not False
        or contract.get("qualified_claim") is not None
        or not isinstance(contract_claims, Mapping)
        or set(contract_claims) != set(EXPECTED_CLAIM_FIELDS)
        or any(value is not None for value in contract_claims.values())
        or not isinstance(contract_gates, Mapping)
        or set(contract_gates) != expected_contract_gates
        or any(value is not True for value in contract_gates.values())
        or not isinstance(parent_checks, Mapping)
        or set(parent_checks) != expected_parent_checks
        or any(value is not True for value in parent_checks.values())
        or not isinstance(contract_bindings, Mapping)
        or set(contract_bindings) != set(expected_contract_bindings)
        or any(
            contract_bindings.get(name) != binding
            for name, binding in expected_contract_bindings.items()
        )
        or contract.get("source_registry_summary", {}).get(
            "source_snapshot_sha256"
        )
        != snapshot["source_snapshot_sha256"]
        or contract.get("source_registry_summary", {}).get(
            "runtime_source_count"
        )
        != snapshot["runtime_source_count"]
        or contract.get("source_registry_summary", {}).get(
            "validation_source_count"
        )
        != snapshot["validation_source_count"]
        or contract.get("source_registry_summary", {}).get(
            "all_registered_sources_live_and_regular"
        )
        is not True
        or live_plan != plan
        or live_seed_registry != seed_registry
        or live_historical_scan != historical_scan
    ):
        raise RuntimeError(
            "V5 outcome-free contract preflight/live bindings are not valid"
        )
    resource_path = root / str(paths["resource_preflight"])
    resource = _strict_json(resource_path)
    resource_consumption = validate_resource_release_evidence(
        root,
        config,
        config_binding,
        plan,
        snapshot,
        resource_path,
        resource,
    )
    resource_binding = _binding(resource_path, root)
    validation = run_focused_validation(
        root,
        config,
        snapshot,
        resource_binding=resource_binding,
        resource_consumption=resource_consumption,
    )
    claims = {field: None for field in EXPECTED_CLAIM_FIELDS}
    gates = {
        "P01_v5_contract_and_live_bindings_pass": True,
        "P02_exact_518_cell_plan": plan["cell_count"] == 518,
        "P03_exact_2085888_rows": plan["row_count"] == 2_085_888,
        "P04_exact_482304_densities": (
            plan["primary_density_count"] == 482_304
        ),
        "P05_full_size_resource_independently_consumed": (
            resource_consumption["ledger_rows_verified"] == 227_328
            and resource_consumption["reset_rows_verified"] == 15_360
            and resource_consumption[
                "formal_seed_addresses_accessed"
            ]
            is False
        ),
        "P06_four_worker_peak_exercised": (
            resource.get("maximum_observed_worker_overlap") == 4
        ),
        "P07_joint_maxt_3037x199_exercised": (
            resource.get("joint_maxt_profile", {}).get("gate_count") == 3037
            and resource.get("joint_maxt_profile", {}).get("replicates") == 199
        ),
        "P08_inventory_finalize_no_copy": (
            resource.get("inventory", {}).get("monolithic_archive") is None
            and resource.get("inventory", {}).get("merged_full_csv") is None
        ),
        "P09_continuous_resource_sampling": (
            int(resource.get("resource_sample_count", 0)) >= 2
        ),
        "P10_source_snapshot_complete": (
            snapshot["runtime_source_count"] == 22
            and snapshot["validation_source_count"] == 9
        ),
        "P11_focused_tests_pass": (
            validation.get("verdict")
            == "PASS_FOCUSED_ANTISIMPLIFICATION"
        ),
        "P12_no_formal_outcome_exists": True,
        "P13_claims_remain_null": True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"preformal release gate failed: {failed}")
    seal: dict[str, Any] = {
        "schema_version": SEAL_SCHEMA,
        "task_id": str(config["task_id"]),
        "verdict": "PASS_PREFORMAL_RELEASE",
        "raw_execution_released": True,
        "scientific_verdict_released": False,
        "formal_outcomes_accessed": False,
        "config_sha256": str(config_binding["sha256"]),
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "source_snapshot": snapshot,
        "bindings": {
            "contract_preflight": _binding(contract_path, root),
            "resource_preflight": _binding(resource_path, root),
            "resource_attempt_ledger": resource_consumption[
                "attempt_chain"
            ]["binding"],
            "resource_start_witness": resource_consumption[
                "attempt_chain"
            ]["start_witness"],
            "resource_pass_witness": resource_consumption[
                "attempt_chain"
            ]["pass_witness"],
            "preformal_validation": _binding(
                root / str(paths["preformal_validation"]),
                root,
            ),
        },
        "resource_run_id": resource["run_id"],
        "resource_consumption": resource_consumption,
        "gates": gates,
        "claim_boundary": claims,
        "scientific_verdict": None,
        "qualified_claim": None,
        "official_puviani_surpass": None,
    }
    seal["analysis_sha256"] = _sha(seal)
    _immutable_json(root / str(paths["preformal_seal"]), seal)
    return seal


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the outcome-blind T04 preformal release seal."
    )
    parser.parse_args(list(argv) if argv is not None else None)
    seal = create_preformal_seal(_root())
    print(json.dumps(seal, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
