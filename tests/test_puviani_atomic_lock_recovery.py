from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from cnn_fpga.benchmark import puviani_paper_constrained_artifacts as artifacts


def _relative_to_repository(path: Path) -> str:
    repository = Path(artifacts.__file__).resolve().parents[2]
    return path.resolve().relative_to(repository.resolve()).as_posix()


def _finalize_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "output_dir": tmp_path / "agents",
        "report_path": tmp_path / "report.json",
        "agent_registry_path": tmp_path / "agents.csv",
        "selection_ledger_path": tmp_path / "selection.csv",
        "training_ledger_path": tmp_path / "training.parquet",
        "trajectory_path": tmp_path / "trajectories.parquet",
        "event_path": tmp_path / "events.parquet",
    }


def _mock_production_launch_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit the isolated recovery-order tests past dedicated launch gates."""

    monkeypatch.setattr(
        artifacts, "_configure_production_determinism", lambda: None
    )
    monkeypatch.setattr(
        artifacts,
        "_validate_runtime_for_config",
        lambda *args, **kwargs: ({"cuda": {"device_uuid": "GPU-lock-test"}}, "r" * 64),
    )
    monkeypatch.setattr(
        artifacts,
        "_validate_gpu_load_attestation",
        lambda *args, **kwargs: {"attestation_sha256": "a" * 64},
    )


def test_only_canonical_finalize_lock_management_bytes_are_excluded(
    tmp_path: Path,
) -> None:
    output = tmp_path / "agents"
    canonical_owner = output / "_locks" / "finalize.lock" / "owner.json"
    canonical_owner.parent.mkdir(parents=True)
    canonical_owner.write_text("lock management", encoding="utf-8")
    direct_orphan = output / "_locks" / "orphan.bin"
    direct_orphan.write_bytes(b"direct orphan")
    nested_orphan = output / "agent" / "_locks" / "forged.bin"
    nested_orphan.parent.mkdir(parents=True)
    nested_orphan.write_bytes(b"nested orphan")

    audit = artifacts._audit_output_tree(
        output_dir=output,
        config={"training": {"paired_root_seeds": []}},
        implementation_hash="test-implementation",
    )
    unknown = set(audit["unknown_or_orphan_files"])
    assert _relative_to_repository(direct_orphan) in unknown
    assert _relative_to_repository(nested_orphan) in unknown
    assert canonical_owner not in audit["output_files"]

    manifest = {"files": []}
    snapshot = artifacts._live_publication_snapshot(manifest, output)
    nested_orphan.write_bytes(b"changed nested orphan")
    with pytest.raises(ValueError, match="changed during validation"):
        artifacts._assert_live_publication_snapshot_unchanged(
            manifest, output, snapshot
        )


def test_competing_finalizer_cannot_recover_before_acquiring_lock(
    tmp_path: Path,
) -> None:
    output = tmp_path / "agents"
    interrupted = output / "_lock_history" / "interrupted"
    interrupted.mkdir(parents=True)
    partial = interrupted / ".recovery.json.host.999.1.tmp"
    partial.write_bytes(b"unsealed recovery evidence")
    before = partial.read_bytes()

    entered = threading.Event()
    release = threading.Event()
    holder_errors: list[BaseException] = []

    def hold_lock() -> None:
        try:
            with artifacts._namespace_finalize_lock(output):
                entered.set()
                if not release.wait(timeout=10.0):
                    raise TimeoutError("test did not release the first finalizer")
        except BaseException as error:  # pragma: no cover - surfaced below.
            holder_errors.append(error)
            entered.set()

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert entered.wait(timeout=10.0)
    try:
        assert holder_errors == []
        with pytest.raises(
            artifacts.ConcurrentAgentWriterError,
            match="live finalizer owns the namespace",
        ):
            with artifacts._namespace_finalize_lock(output):
                raise AssertionError("the competing finalizer must not enter")
        assert partial.read_bytes() == before
        assert not (interrupted / "recovery.json").exists()
    finally:
        release.set()
        holder.join(timeout=10.0)
    assert not holder.is_alive()
    assert holder_errors == []


def test_public_live_validation_does_not_recover_output_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "agents"
    interrupted = output / "_finalization_recovery" / "interrupted"
    interrupted.mkdir(parents=True)
    partial = interrupted / ".report.json.host.999.1.tmp"
    partial.write_bytes(b"unsealed finalization evidence")
    preview = {"output_tree_audit": {"root": _relative_to_repository(output)}}
    monkeypatch.setattr(
        artifacts,
        "_validate_report_locked",
        lambda payload, **kwargs: copy.deepcopy(dict(payload)),
    )

    assert artifacts.validate_report(preview, verify_live_files=True) == preview
    assert partial.read_bytes() == b"unsealed finalization evidence"
    assert not (interrupted / "recovery.json").exists()


def test_finalizer_revokes_old_pass_before_any_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_production_launch_gates(monkeypatch)
    paths = _finalize_paths(tmp_path)
    paths["report_path"].write_text(
        json.dumps({"status": artifacts.STATUS_PASS}), encoding="utf-8"
    )
    interrupted = paths["output_dir"] / "_lock_history" / "interrupted"
    interrupted.mkdir(parents=True)
    (interrupted / "owner.json").write_text("{}", encoding="utf-8")
    observations: list[tuple[str, str]] = []
    real_lock_recovery = artifacts._recover_incomplete_lock_history
    real_finalization_recovery = artifacts._recover_incomplete_finalization_recoveries

    def report_status() -> str:
        return str(
            json.loads(paths["report_path"].read_text(encoding="utf-8"))["status"]
        )

    def recover_lock_history(output: Path) -> None:
        observations.append(("lock_history", report_status()))
        real_lock_recovery(output)

    def recover_finalization_history(output: Path) -> None:
        observations.append(("finalization_history", report_status()))
        real_finalization_recovery(output)

    candidate = {"status": artifacts.STATUS_PASS, "analysis_sha256": "candidate"}
    monkeypatch.setattr(
        artifacts, "_recover_incomplete_lock_history", recover_lock_history
    )
    monkeypatch.setattr(
        artifacts,
        "_recover_incomplete_finalization_recoveries",
        recover_finalization_history,
    )
    monkeypatch.setattr(
        artifacts,
        "_finalize_artifacts_locked",
        lambda *args, **kwargs: copy.deepcopy(candidate),
    )
    monkeypatch.setattr(
        artifacts,
        "_validate_report_locked",
        lambda payload, **kwargs: copy.deepcopy(dict(payload)),
    )

    artifacts.finalize_artifacts({"test": True}, production=True, **paths)
    assert observations == [
        ("lock_history", "INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL"),
        ("finalization_history", "INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL"),
    ]
    assert (interrupted / "recovery.json").is_file()


def test_recovery_failure_retains_only_a_non_pass_failure_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_production_launch_gates(monkeypatch)
    paths = _finalize_paths(tmp_path)
    paths["report_path"].write_text(
        json.dumps({"status": artifacts.STATUS_PASS}), encoding="utf-8"
    )

    def fail_recovery(output: Path) -> None:
        del output
        raise OSError("injected recovery failure")

    monkeypatch.setattr(
        artifacts, "_recover_incomplete_lock_history", fail_recovery
    )
    with pytest.raises(OSError, match="injected recovery failure"):
        artifacts.finalize_artifacts({"test": True}, production=True, **paths)

    marker = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert marker["status"] == "FINALIZATION_FAILED_NO_VALID_SEAL"
    assert marker["valid_pass_seal"] is False
    assert marker["failure_type"] == "OSError"
