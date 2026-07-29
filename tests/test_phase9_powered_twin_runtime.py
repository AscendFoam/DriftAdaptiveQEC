from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from cnn_fpga.benchmark.phase9_powered_twin_bootstrap import (
    THREAD_ENVIRONMENT,
    assert_process_entry_seal,
)
from cnn_fpga.benchmark.phase9_powered_twin_runtime import (
    ActiveOwnerError,
    HeartbeatService,
    OWNER_SCHEMA,
    OwnerLease,
    ResourceLimitExceeded,
    ResourceWatchdog,
    StaleOwnerError,
)


def _lease(path: Path) -> OwnerLease:
    return OwnerLease(
        path,
        run_id="fresh-test-run",
        config_sha256="1" * 64,
        plan_sha256="2" * 64,
    )


def test_process_entry_seal_requires_all_four_vars_before_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    # pytest itself may have imported NumPy; isolate just the policy check by
    # temporarily removing all real numpy/scipy module entries.
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "numpy"
        or name.startswith("numpy.")
        or name == "scipy"
        or name.startswith("scipy.")
    }
    for name in saved:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert_process_entry_seal()
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="thread environment drift"):
        assert_process_entry_seal()
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setitem(sys.modules, "numpy.fake_preimport", object())
    with pytest.raises(RuntimeError, match="imported before"):
        assert_process_entry_seal()


def test_owner_lease_rejects_second_live_owner_and_token_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owner.lock"
    first = _lease(path)
    identity = first.acquire()
    assert identity.pid == os.getpid()
    with pytest.raises(ActiveOwnerError, match="active T04 owner"):
        _lease(path).acquire()
    record = json.loads(path.read_text(encoding="utf-8"))
    record["owner_token"] = "mutated"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="token or lineage drift"):
        first.release()


def test_stale_owner_never_auto_deletes_and_requires_archive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owner.lock"
    stale = {
        "schema_version": OWNER_SCHEMA,
        "host": "dead-host",
        "pid": 2_147_483_000,
        "process_creation_time": 1.0,
        "boot_session": "0" * 64,
        "owner_token": "dead-token",
        "run_id": "fresh-test-run",
        "config_sha256": "1" * 64,
        "plan_sha256": "2" * 64,
    }
    path.write_text(json.dumps(stale), encoding="utf-8")
    lease = _lease(path)
    with pytest.raises(StaleOwnerError, match="explicit archive"):
        lease.acquire()
    assert path.exists()
    archive_path = tmp_path / "stale-owner-archive.json"
    archive = lease.archive_stale(archive_path)
    assert archive["reason"] == "RECORDED_PROCESS_IDENTITY_NOT_LIVE"
    assert archive_path.exists()
    assert not path.exists()
    lease.acquire()
    lease.release()


def test_heartbeat_advances_while_no_chunk_completes(tmp_path: Path) -> None:
    lease = _lease(tmp_path / "owner.lock")
    identity = lease.acquire()
    state = {"active_chunk_ids": ["c44-b-long"], "committed_chunks": 0}
    service = HeartbeatService(
        path=tmp_path / "heartbeat.json",
        owner=lease,
        period_seconds=0.04,
        snapshot=lambda: state,
    )
    service.start()
    time.sleep(0.17)
    service.stop()
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["owner_token"] == identity.owner_token
    assert heartbeat["sequence"] >= 2
    assert heartbeat["snapshot"]["active_chunk_ids"] == ["c44-b-long"]
    assert heartbeat["snapshot"]["committed_chunks"] == 0
    lease.release()


class _FakeProcess:
    def __init__(self, rss: int) -> None:
        self.rss = rss

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self.rss)


def test_resource_watchdog_admits_safe_boundary_and_fails_closed(
    tmp_path: Path,
) -> None:
    values = iter((0.0, 10.0))
    watchdog = ResourceWatchdog(
        run_directory=tmp_path,
        maximum_wall_seconds=1000.0,
        maximum_peak_rss_bytes=1000,
        minimum_post_projection_free_bytes=0,
        maximum_artifact_bytes=1000,
        clock=lambda: next(values),
        process=_FakeProcess(100),
    )
    decision = watchdog.check(
        committed_bytes=100,
        projected_remaining_bytes=100,
        maximum_inflight_temp_bytes=10,
        analysis_scratch_bytes=10,
        projected_remaining_wall_seconds=10,
    )
    assert decision["admit_next_chunk"] is True
    assert decision["failed_limits"] == []

    values = iter((0.0, 1.0))
    blocked = ResourceWatchdog(
        run_directory=tmp_path,
        maximum_wall_seconds=2.0,
        maximum_peak_rss_bytes=50,
        minimum_post_projection_free_bytes=0,
        maximum_artifact_bytes=150,
        clock=lambda: next(values),
        process=_FakeProcess(100),
    )
    with pytest.raises(ResourceLimitExceeded) as captured:
        blocked.check(
            committed_bytes=100,
            projected_remaining_bytes=100,
            maximum_inflight_temp_bytes=0,
            analysis_scratch_bytes=0,
            projected_remaining_wall_seconds=10,
        )
    assert captured.value.decision["admit_next_chunk"] is False
    assert set(captured.value.decision["failed_limits"]) == {
        "wall",
        "rss",
        "artifact",
    }
