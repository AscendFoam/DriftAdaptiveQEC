"""Owner lease, periodic heartbeat and resource watchdog for T04.

These controls are deliberately independent from cell completion.  A long
c44-B chunk therefore continues to emit liveness and resource evidence, while
new work is admitted only at safe chunk boundaries.  Stale owner locks are
never deleted automatically: recovery requires a separately archived proof
that the recorded process identity is no longer live.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import socket
from threading import Event, Lock, Thread
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

import psutil


OWNER_SCHEMA = "PHASE9-POWERED-TWIN-OWNER-LEASE-V1"
HEARTBEAT_SCHEMA = "PHASE9-POWERED-TWIN-HEARTBEAT-V1"
STALE_ARCHIVE_SCHEMA = "PHASE9-POWERED-TWIN-STALE-OWNER-ARCHIVE-V1"
RESOURCE_DECISION_SCHEMA = "PHASE9-POWERED-TWIN-RESOURCE-DECISION-V1"


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    payload = _canonical(value) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _process_creation_time(pid: int) -> float | None:
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def _boot_session() -> str:
    return sha256(
        f"{socket.gethostname()}|{psutil.boot_time():.6f}".encode("utf-8")
    ).hexdigest()


class ActiveOwnerError(RuntimeError):
    pass


class StaleOwnerError(RuntimeError):
    pass


class ResourceLimitExceeded(RuntimeError):
    def __init__(self, decision: Mapping[str, Any]) -> None:
        super().__init__(
            "INCOMPLETE_RESOURCE_FAIL_CLOSED: "
            + ",".join(decision.get("failed_limits", []))
        )
        self.decision = dict(decision)


@dataclass(frozen=True)
class OwnerIdentity:
    schema_version: str
    host: str
    pid: int
    process_creation_time: float
    boot_session: str
    owner_token: str
    run_id: str
    config_sha256: str
    plan_sha256: str


class OwnerLease:
    """Exclusive process-identity lease with explicit stale recovery."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        config_sha256: str,
        plan_sha256: str,
    ) -> None:
        self.path = path.resolve()
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.plan_sha256 = plan_sha256
        self.identity: OwnerIdentity | None = None

    @staticmethod
    def _is_same_live_process(record: Mapping[str, Any]) -> bool:
        pid = record.get("pid")
        created = record.get("process_creation_time")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or not isinstance(created, (int, float))
        ):
            return False
        live = _process_creation_time(pid)
        return live is not None and abs(live - float(created)) <= 1.0e-3

    def acquire(self) -> OwnerIdentity:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            record = _strict_json(self.path)
            if self._is_same_live_process(record):
                raise ActiveOwnerError(
                    f"active T04 owner pid={record.get('pid')} "
                    f"token={record.get('owner_token')}"
                )
            raise StaleOwnerError(
                "stale T04 owner requires explicit archive_stale() evidence"
            )
        pid = os.getpid()
        created = _process_creation_time(pid)
        if created is None:
            raise RuntimeError("cannot attest current process creation time")
        identity = OwnerIdentity(
            schema_version=OWNER_SCHEMA,
            host=platform.node(),
            pid=pid,
            process_creation_time=created,
            boot_session=_boot_session(),
            owner_token=uuid4().hex,
            run_id=self.run_id,
            config_sha256=self.config_sha256,
            plan_sha256=self.plan_sha256,
        )
        payload = _canonical(identity.__dict__) + b"\n"
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise ActiveOwnerError("owner lease race lost") from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            # A partially created current-attempt lock is not silently reused.
            raise
        self.identity = identity
        return identity

    def archive_stale(self, archive_path: Path) -> dict[str, Any]:
        """Archive, but never delete, a proven-dead prior owner record."""

        if not self.path.exists():
            raise FileNotFoundError("owner lock does not exist")
        record = _strict_json(self.path)
        if self._is_same_live_process(record):
            raise ActiveOwnerError("cannot archive a live owner")
        if (
            record.get("schema_version") != OWNER_SCHEMA
            or record.get("run_id") != self.run_id
            or record.get("config_sha256") != self.config_sha256
            or record.get("plan_sha256") != self.plan_sha256
        ):
            raise RuntimeError("stale owner lineage mismatch")
        archive: dict[str, Any] = {
            "schema_version": STALE_ARCHIVE_SCHEMA,
            "reason": "RECORDED_PROCESS_IDENTITY_NOT_LIVE",
            "archived_owner": record,
            "archiver": {
                "host": platform.node(),
                "pid": os.getpid(),
                "process_creation_time": _process_creation_time(os.getpid()),
                "boot_session": _boot_session(),
            },
        }
        archive["analysis_sha256"] = _sha(archive)
        if archive_path.exists():
            if archive_path.read_bytes() != _canonical(archive) + b"\n":
                raise RuntimeError("conflicting stale-owner archive")
        else:
            _atomic_json(archive_path, archive)
        self.path.unlink()
        return archive

    def verify_current(self) -> OwnerIdentity:
        if self.identity is None:
            raise RuntimeError("owner lease has not been acquired")
        record = _strict_json(self.path)
        if record != self.identity.__dict__:
            raise RuntimeError("owner token or lineage drift")
        if not self._is_same_live_process(record):
            raise RuntimeError("owner process identity is no longer live")
        return self.identity

    def release(self) -> None:
        identity = self.verify_current()
        # Release is safe only for the exact current token.  The attempt ledger
        # is responsible for recording the terminal event before this call.
        if _strict_json(self.path).get("owner_token") != identity.owner_token:
            raise RuntimeError("owner token changed before release")
        self.path.unlink()
        self.identity = None


class HeartbeatService:
    """Independent periodic heartbeat; not coupled to chunk completion."""

    def __init__(
        self,
        *,
        path: Path,
        owner: OwnerLease,
        period_seconds: float,
        snapshot: Callable[[], Mapping[str, Any]],
    ) -> None:
        if period_seconds <= 0:
            raise ValueError("heartbeat period must be positive")
        self.path = path.resolve()
        self.owner = owner
        self.period_seconds = float(period_seconds)
        self.snapshot = snapshot
        self._stop = Event()
        self._thread: Thread | None = None
        self._sequence = 0
        self._lock = Lock()
        self.error: BaseException | None = None

    def _write(self) -> None:
        identity = self.owner.verify_current()
        with self._lock:
            value: dict[str, Any] = {
                "schema_version": HEARTBEAT_SCHEMA,
                "run_id": identity.run_id,
                "owner_token": identity.owner_token,
                "pid": identity.pid,
                "process_creation_time": identity.process_creation_time,
                "sequence": self._sequence,
                "monotonic_seconds": time.monotonic(),
                "wall_time_ns": time.time_ns(),
                "snapshot": dict(self.snapshot()),
            }
            value["heartbeat_sha256"] = _sha(value)
            _atomic_json(self.path, value)
            self._sequence += 1

    def _run(self) -> None:
        try:
            self._write()
            while not self._stop.wait(self.period_seconds):
                self._write()
        except BaseException as exc:
            self.error = exc
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("heartbeat already started")
        self._thread = Thread(
            target=self._run,
            name="phase9-t04-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def write_once(self) -> None:
        """Atomically publish a caller-requested stage-boundary heartbeat."""

        if self._thread is None:
            raise RuntimeError("heartbeat has not started")
        if self.error is not None:
            raise RuntimeError("heartbeat service failed") from self.error
        self._write()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 2.0 * self.period_seconds))
            if self._thread.is_alive():
                raise RuntimeError("heartbeat thread did not stop")
        if self.error is not None:
            raise RuntimeError("heartbeat service failed") from self.error


class ResourceWatchdog:
    """Admission control evaluated at safe chunk boundaries."""

    def __init__(
        self,
        *,
        run_directory: Path,
        maximum_wall_seconds: float,
        maximum_peak_rss_bytes: int,
        minimum_post_projection_free_bytes: int,
        maximum_artifact_bytes: int,
        clock: Callable[[], float] = time.monotonic,
        process: psutil.Process | None = None,
    ) -> None:
        self.run_directory = run_directory.resolve()
        self.maximum_wall_seconds = float(maximum_wall_seconds)
        self.maximum_peak_rss_bytes = int(maximum_peak_rss_bytes)
        self.minimum_post_projection_free_bytes = int(
            minimum_post_projection_free_bytes
        )
        self.maximum_artifact_bytes = int(maximum_artifact_bytes)
        self.clock = clock
        self.process = process or psutil.Process(os.getpid())
        self.started = clock()
        self.peak_rss_bytes = 0

    def check(
        self,
        *,
        committed_bytes: int,
        projected_remaining_bytes: int,
        maximum_inflight_temp_bytes: int,
        analysis_scratch_bytes: int,
        projected_remaining_wall_seconds: float,
    ) -> dict[str, Any]:
        rss = int(self.process.memory_info().rss)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        elapsed = float(self.clock() - self.started)
        disk = shutil.disk_usage(self.run_directory)
        projected_artifact = int(committed_bytes) + int(projected_remaining_bytes)
        post_projection_free = (
            int(disk.free)
            - int(projected_remaining_bytes)
            - int(maximum_inflight_temp_bytes)
            - int(analysis_scratch_bytes)
        )
        failed: list[str] = []
        if elapsed + float(projected_remaining_wall_seconds) > self.maximum_wall_seconds:
            failed.append("wall")
        if self.peak_rss_bytes > self.maximum_peak_rss_bytes:
            failed.append("rss")
        if projected_artifact > self.maximum_artifact_bytes:
            failed.append("artifact")
        if post_projection_free < self.minimum_post_projection_free_bytes:
            failed.append("disk")
        decision: dict[str, Any] = {
            "schema_version": RESOURCE_DECISION_SCHEMA,
            "admit_next_chunk": not failed,
            "failed_limits": failed,
            "elapsed_seconds": elapsed,
            "projected_remaining_wall_seconds": float(
                projected_remaining_wall_seconds
            ),
            "current_rss_bytes": rss,
            "peak_rss_bytes": self.peak_rss_bytes,
            "disk_free_bytes": int(disk.free),
            "committed_bytes": int(committed_bytes),
            "projected_remaining_bytes": int(projected_remaining_bytes),
            "maximum_inflight_temp_bytes": int(maximum_inflight_temp_bytes),
            "analysis_scratch_bytes": int(analysis_scratch_bytes),
            "projected_artifact_bytes": projected_artifact,
            "post_projection_free_bytes": post_projection_free,
        }
        decision["decision_sha256"] = _sha(decision)
        if failed:
            raise ResourceLimitExceeded(decision)
        return decision


__all__ = [
    "ActiveOwnerError",
    "HEARTBEAT_SCHEMA",
    "HeartbeatService",
    "OWNER_SCHEMA",
    "OwnerIdentity",
    "OwnerLease",
    "RESOURCE_DECISION_SCHEMA",
    "ResourceLimitExceeded",
    "ResourceWatchdog",
    "STALE_ARCHIVE_SCHEMA",
    "StaleOwnerError",
]
