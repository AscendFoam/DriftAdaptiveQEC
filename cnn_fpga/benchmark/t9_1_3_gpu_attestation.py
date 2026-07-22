"""Fail-closed GPU-load attestation contract for T9.1.3 production launches.

The PowerShell supervisor owns acquisition.  This module independently checks
the raw ``nvidia-smi`` CSV, the parsed rows, load thresholds, canonical hash,
freshness, supervisor parent identity, and the CUDA runtime selected by the
child.  It deliberately has no torch import so synthetic contract tests remain
GPU-free; the caller supplies its already captured runtime signature.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import statistics
from typing import Any, Mapping
import uuid


SCHEMA_VERSION = "t9.1.3-gpu-load-attestation-v1"
BINDING_SCHEMA_VERSION = "t9.1.3-gpu-load-attestation-binding-v1"
TASK_ID = "T9.1.3"
MAX_AGE_SECONDS = 45.0
CLOCK_SKEW_SECONDS = 5.0
PURPOSES = frozenset({"TRAINING_LAUNCH", "FINALIZER_LAUNCH"})
QUERY_ARGUMENTS = (
    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
)
ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "purpose",
        "config_sha256",
        "implementation_sha256",
        "run_identity",
        "attestation_nonce",
        "sampling_started_at_utc",
        "sampling_completed_at_utc",
        "issued_at_utc",
        "expires_at_utc",
        "target_gpu",
        "load_gate",
        "attestation_sha256",
    }
)
BINDING_KEYS = frozenset(
    {
        "schema_version",
        "attestation_sha256",
        "purpose",
        "transaction_id",
        "run_dir",
        "attestation_nonce",
        "target_gpu_uuid",
        "target_gpu_name",
        "target_gpu_total_memory_mib",
        "sampling_completed_at_utc",
        "expires_at_utc",
    }
)
RUN_IDENTITY_KEYS = frozenset(
    {
        "transaction_id",
        "run_dir",
        "supervisor_pid",
        "supervisor_process_created_unix_ns",
        "supervisor_hostname",
    }
)
TARGET_GPU_KEYS = frozenset(
    {"index", "uuid", "name", "memory_total_mib"}
)
LOAD_GATE_KEYS = frozenset(
    {
        "schema_version",
        "passed",
        "failure_reasons",
        "sampled_at_host",
        "cuda_visible_devices",
        "requested_target_gpu_uuid",
        "sample_interval_seconds",
        "thresholds",
        "summary",
        "parsed_samples",
        "raw_samples",
    }
)
THRESHOLD_KEYS = frozenset(
    {
        "expected_sample_count",
        "minimum_free_memory_mib_every_sample",
        "maximum_median_utilization_percent",
        "maximum_peak_utilization_percent",
    }
)
SUMMARY_KEYS = frozenset(
    {
        "parsed_sample_count",
        "consistent_device_count",
        "device_identity_signature",
        "target_selection_basis",
        "target_index",
        "target_uuid",
        "target_name",
        "target_total_memory_mib",
        "target_minimum_free_memory_mib",
        "target_median_utilization_percent",
        "target_maximum_utilization_percent",
        "all_device_summaries",
    }
)
DEVICE_SUMMARY_KEYS = frozenset(
    {
        "index",
        "uuid",
        "name",
        "metric_sample_count",
        "minimum_free_memory_mib",
        "median_utilization_percent",
        "maximum_utilization_percent",
    }
)
PARSED_SAMPLE_KEYS = frozenset({"sequence", "captured_at_utc", "rows"})
DEVICE_ROW_KEYS = frozenset(
    {
        "index",
        "uuid",
        "name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_percent",
    }
)
RAW_SAMPLE_KEYS = frozenset(
    {
        "sequence",
        "captured_at_utc",
        "completed_at_utc",
        "command",
        "arguments",
        "exit_code",
        "stdout",
        "stderr",
        "parse_error",
    }
)


class GpuLoadAttestationError(ValueError):
    """A production launch lacks a complete, fresh, self-consistent gate."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact_mapping(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise GpuLoadAttestationError(f"{label} schema drifted")
    return dict(value)


def _utc_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise GpuLoadAttestationError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GpuLoadAttestationError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GpuLoadAttestationError(f"{label} must carry UTC offset zero")
    return parsed.astimezone(timezone.utc)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GpuLoadAttestationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GpuLoadAttestationError(f"{label} must be finite")
    return result


def _parse_raw_csv(text: Any, sequence: int) -> list[dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        raise GpuLoadAttestationError(f"raw sample {sequence} has empty stdout")
    rows: list[dict[str, Any]] = []
    try:
        csv_rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as error:
        raise GpuLoadAttestationError(f"raw sample {sequence} CSV is invalid") from error
    for raw in csv_rows:
        if len(raw) != 7:
            raise GpuLoadAttestationError(
                f"raw sample {sequence} must contain exactly seven CSV fields"
            )
        try:
            index = int(raw[0].strip())
            numeric = [float(value.strip()) for value in raw[3:]]
        except ValueError as error:
            raise GpuLoadAttestationError(
                f"raw sample {sequence} contains a non-numeric field"
            ) from error
        uuid_value = raw[1].strip()
        name = raw[2].strip()
        if index < 0 or not uuid_value or not name or not all(map(math.isfinite, numeric)):
            raise GpuLoadAttestationError(f"raw sample {sequence} device row is invalid")
        total, used, free, utilization = numeric
        if (
            total <= 0.0
            or used < 0.0
            or free < 0.0
            or used > total + 1.0
            or free > total + 1.0
            or not 0.0 <= utilization <= 100.0
        ):
            raise GpuLoadAttestationError(
                f"raw sample {sequence} contains impossible GPU counters"
            )
        rows.append(
            {
                "index": index,
                "uuid": uuid_value,
                "name": name,
                "memory_total_mib": total,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "utilization_percent": utilization,
            }
        )
    if not rows:
        raise GpuLoadAttestationError(f"raw sample {sequence} has zero GPU rows")
    return rows


def _validated_device_row(value: Any, label: str) -> dict[str, Any]:
    row = _exact_mapping(value, DEVICE_ROW_KEYS, label)
    if isinstance(row["index"], bool) or not isinstance(row["index"], int) or row["index"] < 0:
        raise GpuLoadAttestationError(f"{label} index is invalid")
    if not isinstance(row["uuid"], str) or not row["uuid"]:
        raise GpuLoadAttestationError(f"{label} UUID is invalid")
    if not isinstance(row["name"], str) or not row["name"]:
        raise GpuLoadAttestationError(f"{label} name is invalid")
    for key in (
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_percent",
    ):
        row[key] = _finite_number(row[key], f"{label}.{key}")
    return row


def _selected_target(
    baseline: list[dict[str, Any]], requested_uuid: Any, visible_devices: Any
) -> tuple[dict[str, Any], str]:
    requested = requested_uuid.strip() if isinstance(requested_uuid, str) else ""
    visible = visible_devices.strip() if isinstance(visible_devices, str) else ""
    if requested:
        matches = [row for row in baseline if row["uuid"].lower() == requested.lower()]
        basis = "EXPLICIT_TARGET_GPU_UUID"
    elif visible:
        token = visible.split(",", 1)[0].strip()
        if token.upper().startswith("GPU-"):
            matches = [row for row in baseline if row["uuid"].lower() == token.lower()]
            basis = "CUDA_VISIBLE_DEVICES_FIRST_UUID"
        elif token.isdigit():
            matches = [row for row in baseline if row["index"] == int(token)]
            basis = "CUDA_VISIBLE_DEVICES_FIRST_INDEX"
        else:
            raise GpuLoadAttestationError("CUDA_VISIBLE_DEVICES target is unsupported")
    elif len(baseline) == 1:
        matches = baseline
        basis = "SOLE_NVIDIA_DEVICE"
    else:
        raise GpuLoadAttestationError("multiple GPUs require an explicit target")
    if len(matches) != 1:
        raise GpuLoadAttestationError("target GPU is not uniquely identified")
    if requested and visible:
        visible_match, _ = _selected_target(baseline, None, visible)
        if visible_match["uuid"].lower() != matches[0]["uuid"].lower():
            raise GpuLoadAttestationError(
                "requested target disagrees with CUDA_VISIBLE_DEVICES"
            )
    return matches[0], basis


def _validate_load_gate(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = _exact_mapping(value, LOAD_GATE_KEYS, "load_gate")
    if gate.get("schema_version") != "t9.1.3-nvidia-load-gate-v1":
        raise GpuLoadAttestationError("load_gate version drifted")
    if gate.get("passed") is not True or gate.get("failure_reasons") != []:
        raise GpuLoadAttestationError("load_gate is not an unqualified PASS")
    if gate.get("sample_interval_seconds") != 2:
        raise GpuLoadAttestationError("load_gate interval must be two seconds")
    if not isinstance(gate.get("sampled_at_host"), str) or not gate["sampled_at_host"]:
        raise GpuLoadAttestationError("load_gate host is missing")
    if gate.get("cuda_visible_devices") is not None and not isinstance(
        gate["cuda_visible_devices"], str
    ):
        raise GpuLoadAttestationError("load_gate CUDA visibility is invalid")
    if gate.get("requested_target_gpu_uuid") is not None and not isinstance(
        gate["requested_target_gpu_uuid"], str
    ):
        raise GpuLoadAttestationError("load_gate requested UUID is invalid")

    thresholds = _exact_mapping(gate.get("thresholds"), THRESHOLD_KEYS, "thresholds")
    expected_thresholds = {
        "expected_sample_count": 5,
        "minimum_free_memory_mib_every_sample": 4096.0,
        "maximum_median_utilization_percent": 15.0,
        "maximum_peak_utilization_percent": 30.0,
    }
    if any(
        _finite_number(thresholds[key], f"thresholds.{key}")
        != float(expected_thresholds[key])
        for key in expected_thresholds
    ):
        raise GpuLoadAttestationError("load_gate thresholds drifted")

    parsed_samples = gate.get("parsed_samples")
    raw_samples = gate.get("raw_samples")
    if not isinstance(parsed_samples, list) or len(parsed_samples) != 5:
        raise GpuLoadAttestationError("load_gate requires five parsed samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != 5:
        raise GpuLoadAttestationError("load_gate requires five raw samples")

    normalized_samples: list[dict[str, Any]] = []
    for sequence, (parsed_value, raw_value) in enumerate(
        zip(parsed_samples, raw_samples, strict=True)
    ):
        parsed = _exact_mapping(
            parsed_value, PARSED_SAMPLE_KEYS, f"parsed_samples[{sequence}]"
        )
        raw = _exact_mapping(raw_value, RAW_SAMPLE_KEYS, f"raw_samples[{sequence}]")
        if parsed.get("sequence") != sequence or raw.get("sequence") != sequence:
            raise GpuLoadAttestationError("GPU sample sequence is non-canonical")
        captured = _utc_datetime(parsed.get("captured_at_utc"), "parsed captured_at")
        if raw.get("captured_at_utc") != parsed.get("captured_at_utc"):
            raise GpuLoadAttestationError("raw/parsed capture timestamps disagree")
        completed = _utc_datetime(raw.get("completed_at_utc"), "raw completed_at")
        if completed < captured:
            raise GpuLoadAttestationError("GPU sample completed before capture")
        if (
            raw.get("command") != "nvidia-smi.exe"
            or tuple(raw.get("arguments", ())) != QUERY_ARGUMENTS
            or raw.get("exit_code") != 0
            or raw.get("parse_error") is not None
            or not isinstance(raw.get("stderr"), str)
        ):
            raise GpuLoadAttestationError("raw nvidia-smi command evidence is invalid")
        rows_value = parsed.get("rows")
        if not isinstance(rows_value, list) or not rows_value:
            raise GpuLoadAttestationError("parsed GPU sample has no rows")
        rows = [
            _validated_device_row(row, f"parsed_samples[{sequence}].rows[{index}]")
            for index, row in enumerate(rows_value)
        ]
        if rows != _parse_raw_csv(raw.get("stdout"), sequence):
            raise GpuLoadAttestationError("raw CSV and parsed GPU rows disagree")
        if len({row["index"] for row in rows}) != len(rows) or len(
            {row["uuid"].lower() for row in rows}
        ) != len(rows):
            raise GpuLoadAttestationError("GPU sample contains duplicate identity")
        normalized_samples.append(
            {"sequence": sequence, "captured_at_utc": parsed["captured_at_utc"], "rows": rows}
        )

    baseline = sorted(normalized_samples[0]["rows"], key=lambda row: row["index"])
    identity = [(row["index"], row["uuid"], row["name"], row["memory_total_mib"]) for row in baseline]
    for sample in normalized_samples[1:]:
        current = sorted(sample["rows"], key=lambda row: row["index"])
        if [(row["index"], row["uuid"], row["name"], row["memory_total_mib"]) for row in current] != identity:
            raise GpuLoadAttestationError("GPU identity changed across samples")

    target, selection_basis = _selected_target(
        baseline,
        gate.get("requested_target_gpu_uuid"),
        gate.get("cuda_visible_devices"),
    )
    target_rows = [
        next(row for row in sample["rows"] if row["uuid"].lower() == target["uuid"].lower())
        for sample in normalized_samples
    ]
    free = [row["memory_free_mib"] for row in target_rows]
    utilization = [row["utilization_percent"] for row in target_rows]
    if min(free) < 4096.0 or statistics.median(utilization) > 15.0 or max(utilization) > 30.0:
        raise GpuLoadAttestationError("target GPU does not satisfy the frozen load gate")

    all_device_summaries = []
    for device in baseline:
        rows = [
            next(row for row in sample["rows"] if row["uuid"].lower() == device["uuid"].lower())
            for sample in normalized_samples
        ]
        all_device_summaries.append(
            {
                "index": device["index"],
                "uuid": device["uuid"],
                "name": device["name"],
                "metric_sample_count": 5,
                "minimum_free_memory_mib": min(row["memory_free_mib"] for row in rows),
                "median_utilization_percent": statistics.median(
                    row["utilization_percent"] for row in rows
                ),
                "maximum_utilization_percent": max(
                    row["utilization_percent"] for row in rows
                ),
            }
        )
    expected_summary = {
        "parsed_sample_count": 5,
        "consistent_device_count": len(baseline),
        "device_identity_signature": ";".join(
            f"{row['index']}|{row['uuid']}" for row in baseline
        ),
        "target_selection_basis": selection_basis,
        "target_index": target["index"],
        "target_uuid": target["uuid"],
        "target_name": target["name"],
        "target_total_memory_mib": target["memory_total_mib"],
        "target_minimum_free_memory_mib": min(free),
        "target_median_utilization_percent": statistics.median(utilization),
        "target_maximum_utilization_percent": max(utilization),
        "all_device_summaries": all_device_summaries,
    }
    summary = _exact_mapping(gate.get("summary"), SUMMARY_KEYS, "load_gate.summary")
    for row in summary.get("all_device_summaries", ()):  # exact nested schema first
        _exact_mapping(row, DEVICE_SUMMARY_KEYS, "load_gate.summary.device")
    if summary != expected_summary:
        raise GpuLoadAttestationError("load_gate summary is not derivable from raw samples")
    return gate, target


def seal_gpu_load_attestation(body: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical-seal a supervisor-built body; full validation follows in child."""

    if not isinstance(body, Mapping):
        raise GpuLoadAttestationError("attestation body must be a mapping")
    payload = dict(body)
    if "attestation_sha256" in payload:
        raise GpuLoadAttestationError("unsealed body must not contain attestation_sha256")
    if set(payload) != ATTESTATION_KEYS - {"attestation_sha256"}:
        raise GpuLoadAttestationError("attestation body schema drifted")
    payload["attestation_sha256"] = canonical_sha256(payload)
    return payload


def _runtime_matches_target(runtime: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    if runtime.get("cuda_available") is not True:
        raise GpuLoadAttestationError("attested production child has no CUDA runtime")
    count = runtime.get("cuda_device_count")
    current = runtime.get("cuda_current_device")
    names = runtime.get("cuda_device_names")
    total_bytes = runtime.get("cuda_total_memory_bytes")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(current, bool)
        or not isinstance(current, int)
        or current not in range(count)
        or not isinstance(names, list)
        or len(names) != count
        or not isinstance(total_bytes, list)
        or len(total_bytes) != count
    ):
        raise GpuLoadAttestationError("CUDA runtime device census is invalid")
    if names[current] != target["name"]:
        raise GpuLoadAttestationError("attested GPU name differs from CUDA current device")
    runtime_mib = float(total_bytes[current]) / float(1024**2)
    if abs(runtime_mib - float(target["memory_total_mib"])) > 1.0:
        raise GpuLoadAttestationError("attested GPU memory differs from CUDA current device")
    driver_rows = runtime.get("nvidia_smi_devices")
    if not isinstance(driver_rows, list) or sum(
        isinstance(row, Mapping)
        and str(row.get("uuid", "")).lower() == str(target["uuid"]).lower()
        for row in driver_rows
    ) != 1:
        raise GpuLoadAttestationError("attested GPU UUID is absent or duplicated at runtime")
    controls = runtime.get("environment_controls")
    visible = controls.get("CUDA_VISIBLE_DEVICES") if isinstance(controls, Mapping) else None
    if isinstance(visible, str) and visible.strip():
        token = visible.split(",", 1)[0].strip()
        if token.upper().startswith("GPU-") and token.lower() != str(target["uuid"]).lower():
            raise GpuLoadAttestationError("attested UUID differs from CUDA_VISIBLE_DEVICES")
        if token.isdigit() and int(token) != int(target["index"]):
            raise GpuLoadAttestationError("attested index differs from CUDA_VISIBLE_DEVICES")
    elif len(driver_rows) != 1:
        raise GpuLoadAttestationError(
            "multiple physical GPUs require CUDA_VISIBLE_DEVICES to bind the target"
        )


def validate_gpu_load_attestation(
    value: Mapping[str, Any] | str | Path | None,
    *,
    config_sha256: str,
    implementation_sha256: str,
    expected_purpose: str,
    current_runtime: Mapping[str, Any] | None,
    require_fresh: bool,
    require_live_parent: bool,
    now: datetime | None = None,
    observed_parent_pid: int | None = None,
    observed_parent_created_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Validate one canonical production launch proof and return a normalized copy."""

    if value is None:
        raise GpuLoadAttestationError("production requires --gpu-attestation")
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            payload_value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise GpuLoadAttestationError("GPU attestation file is unreadable") from error
    else:
        payload_value = value
    payload = _exact_mapping(payload_value, ATTESTATION_KEYS, "GPU attestation")
    claimed_hash = payload.get("attestation_sha256")
    unhashed = dict(payload)
    unhashed.pop("attestation_sha256")
    if not _lower_sha256(claimed_hash) or claimed_hash != canonical_sha256(unhashed):
        raise GpuLoadAttestationError("GPU attestation canonical hash mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("task_id") != TASK_ID:
        raise GpuLoadAttestationError("GPU attestation identity mismatch")
    if expected_purpose not in PURPOSES or payload.get("purpose") != expected_purpose:
        raise GpuLoadAttestationError("GPU attestation purpose mismatch")
    if (
        payload.get("config_sha256") != config_sha256
        or payload.get("implementation_sha256") != implementation_sha256
    ):
        raise GpuLoadAttestationError("GPU attestation config/implementation binding mismatch")

    identity = _exact_mapping(payload.get("run_identity"), RUN_IDENTITY_KEYS, "run_identity")
    try:
        uuid.UUID(str(identity["transaction_id"]))
        uuid.UUID(str(payload["attestation_nonce"]))
    except (ValueError, AttributeError) as error:
        raise GpuLoadAttestationError("GPU attestation nonce/transaction is invalid") from error
    run_dir = identity.get("run_dir")
    if not isinstance(run_dir, str) or not Path(run_dir).is_absolute():
        raise GpuLoadAttestationError("GPU attestation run_dir must be absolute")
    if not isinstance(identity.get("supervisor_hostname"), str) or not identity["supervisor_hostname"]:
        raise GpuLoadAttestationError("GPU attestation supervisor hostname is invalid")
    supervisor_pid = identity.get("supervisor_pid")
    supervisor_created = identity.get("supervisor_process_created_unix_ns")
    if (
        isinstance(supervisor_pid, bool)
        or not isinstance(supervisor_pid, int)
        or supervisor_pid <= 0
        or isinstance(supervisor_created, bool)
        or not isinstance(supervisor_created, int)
        or supervisor_created <= 0
    ):
        raise GpuLoadAttestationError("GPU attestation supervisor process identity is invalid")
    if require_live_parent:
        parent_pid = os.getppid() if observed_parent_pid is None else observed_parent_pid
        if parent_pid != supervisor_pid:
            raise GpuLoadAttestationError("production child is not owned by attested supervisor")
        if observed_parent_created_unix_ns is None:
            try:
                import psutil

                observed_parent_created_unix_ns = int(
                    round(float(psutil.Process(parent_pid).create_time()) * 1_000_000_000)
                )
            except (ImportError, OSError, ValueError) as error:
                raise GpuLoadAttestationError(
                    "attested supervisor creation identity is unavailable"
                ) from error
            except Exception as error:  # psutil's platform exception hierarchy
                raise GpuLoadAttestationError(
                    "attested supervisor is not live"
                ) from error
        if abs(int(observed_parent_created_unix_ns) - supervisor_created) > 1_000_000:
            raise GpuLoadAttestationError("attested supervisor PID was reused or forged")
        if identity["supervisor_hostname"] != socket.gethostname():
            raise GpuLoadAttestationError("attested supervisor host differs from child host")

    gate, derived_target = _validate_load_gate(payload.get("load_gate"))
    target = _exact_mapping(payload.get("target_gpu"), TARGET_GPU_KEYS, "target_gpu")
    normalized_target = {
        "index": target.get("index"),
        "uuid": target.get("uuid"),
        "name": target.get("name"),
        "memory_total_mib": _finite_number(
            target.get("memory_total_mib"), "target_gpu.memory_total_mib"
        ),
    }
    if normalized_target != {
        key: derived_target[key] for key in ("index", "uuid", "name", "memory_total_mib")
    }:
        raise GpuLoadAttestationError("target_gpu is not derived from raw load samples")
    if gate["sampled_at_host"] != identity["supervisor_hostname"]:
        raise GpuLoadAttestationError("load sample host differs from run identity")
    if current_runtime is not None:
        _runtime_matches_target(current_runtime, normalized_target)

    sampling_started = _utc_datetime(payload.get("sampling_started_at_utc"), "sampling_started_at_utc")
    sampling_completed = _utc_datetime(payload.get("sampling_completed_at_utc"), "sampling_completed_at_utc")
    issued = _utc_datetime(payload.get("issued_at_utc"), "issued_at_utc")
    expires = _utc_datetime(payload.get("expires_at_utc"), "expires_at_utc")
    raw_samples = gate["raw_samples"]
    if (
        payload["sampling_started_at_utc"] != raw_samples[0]["captured_at_utc"]
        or payload["sampling_completed_at_utc"] != raw_samples[-1]["completed_at_utc"]
        or not sampling_started <= sampling_completed <= issued < expires
        or (issued - sampling_completed).total_seconds() > CLOCK_SKEW_SECONDS
        or (expires - issued).total_seconds() > MAX_AGE_SECONDS
    ):
        raise GpuLoadAttestationError("GPU attestation sampling/expiry timeline is invalid")
    if require_fresh:
        observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (observed_now - sampling_completed).total_seconds()
        if age < -CLOCK_SKEW_SECONDS or age > MAX_AGE_SECONDS or observed_now > expires:
            raise GpuLoadAttestationError("GPU attestation is missing freshness or expired")
    return payload


def gpu_attestation_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact immutable fields copied through every evidence layer."""

    payload = _exact_mapping(value, ATTESTATION_KEYS, "GPU attestation")
    identity = _exact_mapping(payload["run_identity"], RUN_IDENTITY_KEYS, "run_identity")
    target = _exact_mapping(payload["target_gpu"], TARGET_GPU_KEYS, "target_gpu")
    binding = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "attestation_sha256": payload["attestation_sha256"],
        "purpose": payload["purpose"],
        "transaction_id": identity["transaction_id"],
        "run_dir": identity["run_dir"],
        "attestation_nonce": payload["attestation_nonce"],
        "target_gpu_uuid": target["uuid"],
        "target_gpu_name": target["name"],
        "target_gpu_total_memory_mib": target["memory_total_mib"],
        "sampling_completed_at_utc": payload["sampling_completed_at_utc"],
        "expires_at_utc": payload["expires_at_utc"],
    }
    if set(binding) != BINDING_KEYS:
        raise AssertionError("internal GPU binding schema drifted")
    return binding


def validate_gpu_attestation_binding(value: Any) -> dict[str, Any]:
    binding = _exact_mapping(value, BINDING_KEYS, "GPU attestation binding")
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise GpuLoadAttestationError("GPU attestation binding version drifted")
    if not _lower_sha256(binding.get("attestation_sha256")):
        raise GpuLoadAttestationError("GPU attestation binding hash is invalid")
    if binding.get("purpose") not in PURPOSES:
        raise GpuLoadAttestationError("GPU attestation binding purpose is invalid")
    for name in (
        "transaction_id",
        "run_dir",
        "attestation_nonce",
        "target_gpu_uuid",
        "target_gpu_name",
        "sampling_completed_at_utc",
        "expires_at_utc",
    ):
        if not isinstance(binding.get(name), str) or not binding[name]:
            raise GpuLoadAttestationError(f"GPU attestation binding {name} is invalid")
    _finite_number(
        binding.get("target_gpu_total_memory_mib"),
        "GPU attestation binding target memory",
    )
    return binding


def synthetic_attestation_self_test() -> dict[str, Any]:
    """Exercise both launch roles and fail cases without querying a real GPU."""

    base_time = datetime.now(timezone.utc)
    parsed_samples: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    utilization = (5.0, 10.0, 15.0, 10.0, 5.0)
    for sequence, util in enumerate(utilization):
        captured = base_time.replace(microsecond=0)
        captured = captured.fromtimestamp(
            captured.timestamp() + 2 * sequence, tz=timezone.utc
        ).isoformat()
        completed = datetime.fromtimestamp(
            datetime.fromisoformat(captured).timestamp() + 0.1,
            tz=timezone.utc,
        ).isoformat()
        row = {
            "index": 0,
            "uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "name": "Synthetic GPU",
            "memory_total_mib": 8192.0,
            "memory_used_mib": 1900.0,
            "memory_free_mib": 6292.0,
            "utilization_percent": util,
        }
        stdout = (
            f"0, {row['uuid']}, {row['name']}, 8192, 1900, 6292, {util}\n"
        )
        parsed_samples.append(
            {"sequence": sequence, "captured_at_utc": captured, "rows": [row]}
        )
        raw_samples.append(
            {
                "sequence": sequence,
                "captured_at_utc": captured,
                "completed_at_utc": completed,
                "command": "nvidia-smi.exe",
                "arguments": list(QUERY_ARGUMENTS),
                "exit_code": 0,
                "stdout": stdout,
                "stderr": "",
                "parse_error": None,
            }
        )
    gate = {
        "schema_version": "t9.1.3-nvidia-load-gate-v1",
        "passed": True,
        "failure_reasons": [],
        "sampled_at_host": socket.gethostname(),
        "cuda_visible_devices": None,
        "requested_target_gpu_uuid": None,
        "sample_interval_seconds": 2,
        "thresholds": {
            "expected_sample_count": 5,
            "minimum_free_memory_mib_every_sample": 4096.0,
            "maximum_median_utilization_percent": 15.0,
            "maximum_peak_utilization_percent": 30.0,
        },
        "summary": {
            "parsed_sample_count": 5,
            "consistent_device_count": 1,
            "device_identity_signature": (
                "0|GPU-00000000-0000-0000-0000-000000000001"
            ),
            "target_selection_basis": "SOLE_NVIDIA_DEVICE",
            "target_index": 0,
            "target_uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "target_name": "Synthetic GPU",
            "target_total_memory_mib": 8192.0,
            "target_minimum_free_memory_mib": 6292.0,
            "target_median_utilization_percent": 10.0,
            "target_maximum_utilization_percent": 15.0,
            "all_device_summaries": [
                {
                    "index": 0,
                    "uuid": "GPU-00000000-0000-0000-0000-000000000001",
                    "name": "Synthetic GPU",
                    "metric_sample_count": 5,
                    "minimum_free_memory_mib": 6292.0,
                    "median_utilization_percent": 10.0,
                    "maximum_utilization_percent": 15.0,
                }
            ],
        },
        "parsed_samples": parsed_samples,
        "raw_samples": raw_samples,
    }

    def make(purpose: str) -> dict[str, Any]:
        sampling_completed = datetime.fromisoformat(
            raw_samples[-1]["completed_at_utc"]
        )
        issued = sampling_completed.fromtimestamp(
            sampling_completed.timestamp() + 0.1, tz=timezone.utc
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "purpose": purpose,
            "config_sha256": "1" * 64,
            "implementation_sha256": "2" * 64,
            "run_identity": {
                "transaction_id": "00000000-0000-0000-0000-000000000003",
                "run_dir": str(Path.cwd().resolve()),
                "supervisor_pid": 123,
                "supervisor_process_created_unix_ns": 456,
                "supervisor_hostname": socket.gethostname(),
            },
            "attestation_nonce": "00000000-0000-0000-0000-000000000004",
            "sampling_started_at_utc": raw_samples[0]["captured_at_utc"],
            "sampling_completed_at_utc": raw_samples[-1]["completed_at_utc"],
            "issued_at_utc": issued.isoformat(),
            "expires_at_utc": datetime.fromtimestamp(
                issued.timestamp() + MAX_AGE_SECONDS, tz=timezone.utc
            ).isoformat(),
            "target_gpu": {
                "index": 0,
                "uuid": "GPU-00000000-0000-0000-0000-000000000001",
                "name": "Synthetic GPU",
                "memory_total_mib": 8192.0,
            },
            "load_gate": gate,
        }
        return seal_gpu_load_attestation(body)

    def accepted(payload: Any, purpose: str, observed_now: datetime) -> bool:
        try:
            validate_gpu_load_attestation(
                payload,
                config_sha256="1" * 64,
                implementation_sha256="2" * 64,
                expected_purpose=purpose,
                current_runtime=None,
                require_fresh=True,
                require_live_parent=False,
                now=observed_now,
            )
        except GpuLoadAttestationError:
            return False
        return True

    training = make("TRAINING_LAUNCH")
    finalizer = make("FINALIZER_LAUNCH")
    fresh_now = datetime.fromisoformat(training["issued_at_utc"])
    missing_rejected = not accepted(None, "TRAINING_LAUNCH", fresh_now)
    stale_now = datetime.fromtimestamp(
        datetime.fromisoformat(training["expires_at_utc"]).timestamp() + 1.0,
        tz=timezone.utc,
    )
    stale_rejected = not accepted(training, "TRAINING_LAUNCH", stale_now)
    tampered = json.loads(json.dumps(training))
    tampered["load_gate"]["parsed_samples"][0]["rows"][0][
        "memory_free_mib"
    ] = 7000.0
    tamper_rejected = not accepted(tampered, "TRAINING_LAUNCH", fresh_now)
    wrong_uuid = json.loads(json.dumps(training))
    wrong_uuid["target_gpu"]["uuid"] = (
        "GPU-00000000-0000-0000-0000-000000000099"
    )
    wrong_uuid.pop("attestation_sha256")
    wrong_uuid = seal_gpu_load_attestation(wrong_uuid)
    uuid_rejected = not accepted(wrong_uuid, "TRAINING_LAUNCH", fresh_now)
    training_pass = accepted(training, "TRAINING_LAUNCH", fresh_now)
    finalizer_pass = accepted(finalizer, "FINALIZER_LAUNCH", fresh_now)
    purpose_swap_rejected = not accepted(
        training, "FINALIZER_LAUNCH", fresh_now
    )
    result = {
        "schema_version": "t9.1.3-gpu-attestation-static-self-test-v1",
        "training_gate_pass": training_pass,
        "finalizer_gate_pass": finalizer_pass,
        "missing_rejected": missing_rejected,
        "stale_rejected": stale_rejected,
        "tamper_rejected": tamper_rejected,
        "uuid_mismatch_rejected": uuid_rejected,
        "purpose_swap_rejected": purpose_swap_rejected,
        "gpu_queried": False,
        "production_started": False,
    }
    result["status"] = (
        "PASS"
        if all(
            result[name] is True
            for name in (
                "training_gate_pass",
                "finalizer_gate_pass",
                "missing_rejected",
                "stale_rejected",
                "tamper_rejected",
                "uuid_mismatch_rejected",
                "purpose_swap_rejected",
            )
        )
        else "FAIL"
    )
    return result
