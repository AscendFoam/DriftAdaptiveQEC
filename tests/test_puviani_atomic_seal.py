from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cnn_fpga.benchmark import puviani_paper_constrained_artifacts as artifacts


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


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
    """Keep atomic-publication tests focused after strict launch gates pass.

    Runtime-ledger and GPU-attestation rejection paths have dedicated tests;
    these tests exercise the namespace mutation and publication ordering that
    follows successful production admission.
    """

    runtime = {"cuda": {"device_uuid": "GPU-atomic-test"}}
    monkeypatch.setattr(
        artifacts, "_configure_production_determinism", lambda: None
    )
    monkeypatch.setattr(
        artifacts,
        "_validate_runtime_for_config",
        lambda *args, **kwargs: (copy.deepcopy(runtime), "runtime-sha256"),
    )
    monkeypatch.setattr(
        artifacts,
        "_validate_gpu_load_attestation",
        lambda *args, **kwargs: {"attestation_sha256": "a" * 64},
    )


def test_finalize_invalidates_old_pass_before_build_and_validates_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_production_launch_gates(monkeypatch)
    paths = _finalize_paths(tmp_path)
    paths["report_path"].write_text(
        json.dumps({"status": artifacts.STATUS_PASS}), encoding="utf-8"
    )
    observations: list[str] = []
    candidate = {"status": artifacts.STATUS_PASS, "analysis_sha256": "candidate"}

    def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        observations.append(_read_json(paths["report_path"])["status"])
        return copy.deepcopy(candidate)

    def validate(payload: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["verify_live_files"] is True
        assert payload == candidate
        observations.append(_read_json(paths["report_path"])["status"])
        return dict(payload)

    monkeypatch.setattr(artifacts, "_finalize_artifacts_locked", build)
    monkeypatch.setattr(artifacts, "_validate_report_locked", validate)
    result = artifacts.finalize_artifacts(
        {"test": True}, production=True, gpu_attestation={"test": True}, **paths
    )

    assert observations == [
        "INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL",
        "INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL",
    ]
    assert result == candidate
    assert _read_json(paths["report_path"])["status"] == artifacts.STATUS_PASS


def test_live_validation_failure_never_leaves_pass_and_retains_invalid_forensic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_production_launch_gates(monkeypatch)
    paths = _finalize_paths(tmp_path)
    paths["report_path"].write_text(
        json.dumps({"status": artifacts.STATUS_PASS}), encoding="utf-8"
    )
    candidate = {"status": artifacts.STATUS_PASS, "analysis_sha256": "candidate"}
    monkeypatch.setattr(
        artifacts,
        "_finalize_artifacts_locked",
        lambda *args, **kwargs: copy.deepcopy(candidate),
    )

    def reject(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise ValueError("injected live-file mismatch")

    monkeypatch.setattr(artifacts, "_validate_report_locked", reject)
    with pytest.raises(ValueError, match="injected live-file mismatch"):
        artifacts.finalize_artifacts(
            {"test": True},
            production=True,
            gpu_attestation={"test": True},
            **paths,
        )

    marker = _read_json(paths["report_path"])
    assert marker["status"] == "FINALIZATION_FAILED_NO_VALID_SEAL"
    assert marker["valid_pass_seal"] is False
    forensic_path = Path(marker["forensic_candidate"]["path"])
    if not forensic_path.is_absolute():
        forensic_path = Path(artifacts.__file__).resolve().parents[2] / forensic_path
    forensic = _read_json(forensic_path)
    assert forensic["status"] == (
        "INVALID_UNPUBLISHED_CANDIDATE_LIVE_VALIDATION_FAILED"
    )
    assert forensic["valid_pass_seal"] is False
    assert forensic["forensic_validation_failure"]["failure_type"] == "ValueError"


def test_public_live_validation_uses_same_exclusive_namespace_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "agents"
    output.mkdir(parents=True)
    repository = Path(artifacts.__file__).resolve().parents[2]
    relative_output = output.resolve().relative_to(repository.resolve()).as_posix()
    preview = {"output_tree_audit": {"root": relative_output}}
    monkeypatch.setattr(
        artifacts,
        "_validate_report_locked",
        lambda payload, **kwargs: dict(payload),
    )

    with artifacts._namespace_finalize_lock(output):
        with pytest.raises(
            artifacts.ConcurrentAgentWriterError,
            match="live finalizer owns the namespace",
        ):
            artifacts.validate_report(preview, verify_live_files=True)


def test_report_change_during_locked_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "agents"
    output.mkdir(parents=True)
    repository = Path(artifacts.__file__).resolve().parents[2]
    preview = {
        "output_tree_audit": {
            "root": output.resolve().relative_to(repository.resolve()).as_posix()
        },
        "sentinel": "sealed",
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps(preview), encoding="utf-8")

    def mutate_report(payload: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        changed = copy.deepcopy(dict(payload))
        changed["sentinel"] = "changed"
        report.write_text(json.dumps(changed), encoding="utf-8")
        return dict(payload)

    monkeypatch.setattr(artifacts, "_validate_report_locked", mutate_report)
    with pytest.raises(ValueError, match="report changed during locked"):
        artifacts.validate_report(report, verify_live_files=True)


def test_manifest_or_output_change_mid_validation_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "agents"
    output.mkdir(parents=True)
    evidence = output / "evidence.bin"
    evidence.write_bytes(b"sealed")
    repository = Path(artifacts.__file__).resolve().parents[2]
    relative = evidence.resolve().relative_to(repository.resolve()).as_posix()
    manifest = {
        "files": [
            {
                "path": relative,
                "bytes": evidence.stat().st_size,
                "sha256": artifacts._file_sha256(evidence),
            }
        ]
    }
    initial = artifacts._live_publication_snapshot(manifest, output)
    (output / "injected.bin").write_bytes(b"changed during validation")

    with pytest.raises(ValueError, match="changed during validation"):
        artifacts._assert_live_publication_snapshot_unchanged(
            manifest, output, initial
        )


def _raw_audit(config: dict[str, Any]) -> dict[str, Any]:
    qualification = config["six_state_qualification"]
    group_count = 3 * len(artifacts.STATE_LABELS) * (
        len(qualification["seeds"]) + len(qualification["confirmation_seeds"])
    )
    return {
        "schema_version": "t9.1.3-six-state-same-backend-replay-v1",
        "scope": (
            "seeded same-runtime/same-backend replay of selected checkpoints, RNG, "
            "physics, actions and raw rows; not cross-runtime bitwise reproducibility "
            "and not an independent second physics backend"
        ),
        "group_count": group_count,
        "trajectory_count": 1008,
        "event_count": 20160,
        "maximum_action_absolute_error": 2.0e-15,
        "maximum_branch_probability_absolute_error": 3.0e-15,
        "maximum_projected_rho_absolute_error": 8.0e-14,
        "all_rows_replayed": True,
    }


def test_raw_replay_maxima_allow_tiny_float_drift_but_reject_material_error() -> None:
    config = {
        "six_state_qualification": {
            "seeds": list(range(5)),
            "confirmation_seeds": list(range(2)),
        }
    }
    sealed = _raw_audit(config)
    tiny_drift = copy.deepcopy(sealed)
    tiny_drift["maximum_action_absolute_error"] += 2.0e-15
    tiny_drift["maximum_branch_probability_absolute_error"] += 2.0e-15
    tiny_drift["maximum_projected_rho_absolute_error"] += 1.0e-13
    assert artifacts._six_state_raw_replay_audits_compatible(
        config,
        sealed,
        tiny_drift,
        trajectory_count=1008,
        event_count=20160,
    )

    material = copy.deepcopy(sealed)
    material["maximum_projected_rho_absolute_error"] = 2.0e-11
    assert not artifacts._six_state_raw_replay_audits_compatible(
        config,
        sealed,
        material,
        trajectory_count=1008,
        event_count=20160,
    )

    wrong_count = copy.deepcopy(sealed)
    wrong_count["event_count"] -= 1
    assert not artifacts._six_state_raw_replay_audits_compatible(
        config,
        sealed,
        wrong_count,
        trajectory_count=1008,
        event_count=20160,
    )
