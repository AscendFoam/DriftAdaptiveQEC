from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from cnn_fpga.benchmark import puviani_paper_constrained_artifacts as subject
from cnn_fpga.benchmark import t9_1_3_gpu_attestation as attestation


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase9/t9_1_3_puviani_paper_constrained.json"


def _runtime(uuid_value: str = "GPU-00000000-0000-0000-0000-000000000001") -> dict[str, object]:
    return {
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_current_device": 0,
        "cuda_device_names": ["Synthetic GPU"],
        "cuda_total_memory_bytes": [8192 * 1024**2],
        "nvidia_smi_devices": [{"uuid": uuid_value, "driver_version": "999.0"}],
        "environment_controls": {"CUDA_VISIBLE_DEVICES": uuid_value},
    }


def _attestation(purpose: str = "TRAINING_LAUNCH") -> dict[str, object]:
    completed = datetime.now(timezone.utc) - timedelta(seconds=1)
    started = completed - timedelta(seconds=8.1)
    parsed = []
    raw = []
    utilization = (5.0, 10.0, 15.0, 10.0, 5.0)
    uuid_value = "GPU-00000000-0000-0000-0000-000000000001"
    for sequence, value in enumerate(utilization):
        captured = started + timedelta(seconds=2 * sequence)
        sample_completed = captured + timedelta(milliseconds=100)
        row = {
            "index": 0,
            "uuid": uuid_value,
            "name": "Synthetic GPU",
            "memory_total_mib": 8192.0,
            "memory_used_mib": 1900.0,
            "memory_free_mib": 6292.0,
            "utilization_percent": value,
        }
        parsed.append(
            {
                "sequence": sequence,
                "captured_at_utc": captured.isoformat(),
                "rows": [row],
            }
        )
        raw.append(
            {
                "sequence": sequence,
                "captured_at_utc": captured.isoformat(),
                "completed_at_utc": sample_completed.isoformat(),
                "command": "nvidia-smi.exe",
                "arguments": list(attestation.QUERY_ARGUMENTS),
                "exit_code": 0,
                "stdout": (
                    f"0, {uuid_value}, Synthetic GPU, 8192, 1900, 6292, {value}\n"
                ),
                "stderr": "",
                "parse_error": None,
            }
        )
    issued = datetime.fromisoformat(raw[-1]["completed_at_utc"]) + timedelta(
        milliseconds=100
    )
    gate = {
        "schema_version": "t9.1.3-nvidia-load-gate-v1",
        "passed": True,
        "failure_reasons": [],
        "sampled_at_host": socket.gethostname(),
        "cuda_visible_devices": uuid_value,
        "requested_target_gpu_uuid": uuid_value,
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
            "device_identity_signature": f"0|{uuid_value}",
            "target_selection_basis": "EXPLICIT_TARGET_GPU_UUID",
            "target_index": 0,
            "target_uuid": uuid_value,
            "target_name": "Synthetic GPU",
            "target_total_memory_mib": 8192.0,
            "target_minimum_free_memory_mib": 6292.0,
            "target_median_utilization_percent": 10.0,
            "target_maximum_utilization_percent": 15.0,
            "all_device_summaries": [
                {
                    "index": 0,
                    "uuid": uuid_value,
                    "name": "Synthetic GPU",
                    "metric_sample_count": 5,
                    "minimum_free_memory_mib": 6292.0,
                    "median_utilization_percent": 10.0,
                    "maximum_utilization_percent": 15.0,
                }
            ],
        },
        "parsed_samples": parsed,
        "raw_samples": raw,
    }
    return attestation.seal_gpu_load_attestation(
        {
            "schema_version": attestation.SCHEMA_VERSION,
            "task_id": subject.TASK_ID,
            "purpose": purpose,
            "config_sha256": "1" * 64,
            "implementation_sha256": "2" * 64,
            "run_identity": {
                "transaction_id": "00000000-0000-0000-0000-000000000003",
                "run_dir": str(ROOT.resolve()),
                "supervisor_pid": 123,
                "supervisor_process_created_unix_ns": 456,
                "supervisor_hostname": socket.gethostname(),
            },
            "attestation_nonce": "00000000-0000-0000-0000-000000000004",
            "sampling_started_at_utc": raw[0]["captured_at_utc"],
            "sampling_completed_at_utc": raw[-1]["completed_at_utc"],
            "issued_at_utc": issued.isoformat(),
            "expires_at_utc": (
                issued + timedelta(seconds=attestation.MAX_AGE_SECONDS)
            ).isoformat(),
            "target_gpu": {
                "index": 0,
                "uuid": uuid_value,
                "name": "Synthetic GPU",
                "memory_total_mib": 8192.0,
            },
            "load_gate": gate,
        }
    )


def _validate(value: object, *, purpose: str = "TRAINING_LAUNCH", now: datetime | None = None) -> dict[str, object]:
    return attestation.validate_gpu_load_attestation(
        value,
        config_sha256="1" * 64,
        implementation_sha256="2" * 64,
        expected_purpose=purpose,
        current_runtime=_runtime(),
        require_fresh=True,
        require_live_parent=True,
        now=now,
        observed_parent_pid=123,
        observed_parent_created_unix_ns=456,
    )


def test_canonical_training_and_finalizer_attestations_pass_and_bind() -> None:
    training = _attestation()
    finalizer = _attestation("FINALIZER_LAUNCH")
    assert _validate(training) == training
    assert _validate(finalizer, purpose="FINALIZER_LAUNCH") == finalizer
    binding = attestation.gpu_attestation_binding(training)
    assert binding["attestation_sha256"] == training["attestation_sha256"]
    assert binding["target_gpu_uuid"] == training["target_gpu"]["uuid"]
    assert set(binding) == attestation.BINDING_KEYS


def test_windows_hostname_case_variation_is_same_host_but_other_host_is_rejected() -> None:
    payload = deepcopy(_attestation())
    case_variant = socket.gethostname().swapcase()
    payload["run_identity"]["supervisor_hostname"] = case_variant
    payload["load_gate"]["sampled_at_host"] = case_variant
    payload.pop("attestation_sha256")
    payload = attestation.seal_gpu_load_attestation(payload)
    assert _validate(payload) == payload

    other = deepcopy(payload)
    other["run_identity"]["supervisor_hostname"] = "definitely-another-host"
    other["load_gate"]["sampled_at_host"] = "definitely-another-host"
    other.pop("attestation_sha256")
    other = attestation.seal_gpu_load_attestation(other)
    with pytest.raises(attestation.GpuLoadAttestationError, match="host differs"):
        _validate(other)


@pytest.mark.parametrize("failure", ("missing", "stale", "tamper", "uuid", "parent", "purpose"))
def test_missing_stale_tampered_uuid_parent_and_purpose_fail_closed(failure: str) -> None:
    payload: object = _attestation()
    kwargs: dict[str, object] = {}
    if failure == "missing":
        payload = None
    elif failure == "stale":
        kwargs["now"] = datetime.fromisoformat(payload["expires_at_utc"]) + timedelta(seconds=1)
    elif failure == "tamper":
        payload = deepcopy(payload)
        payload["load_gate"]["parsed_samples"][0]["rows"][0]["memory_free_mib"] = 7000.0
    elif failure == "uuid":
        payload = deepcopy(payload)
        payload["target_gpu"]["uuid"] = "GPU-00000000-0000-0000-0000-000000000099"
        payload.pop("attestation_sha256")
        payload = attestation.seal_gpu_load_attestation(payload)
    elif failure == "parent":
        with pytest.raises(attestation.GpuLoadAttestationError, match="supervisor"):
            attestation.validate_gpu_load_attestation(
                payload,
                config_sha256="1" * 64,
                implementation_sha256="2" * 64,
                expected_purpose="TRAINING_LAUNCH",
                current_runtime=_runtime(),
                require_fresh=True,
                require_live_parent=True,
                observed_parent_pid=999,
                observed_parent_created_unix_ns=456,
            )
        return
    else:
        kwargs["purpose"] = "FINALIZER_LAUNCH"
    with pytest.raises(attestation.GpuLoadAttestationError):
        _validate(payload, **kwargs)


def test_runtime_uuid_mismatch_is_rejected_even_when_attestation_is_resealed() -> None:
    payload = _attestation()
    with pytest.raises(attestation.GpuLoadAttestationError, match="UUID"):
        attestation.validate_gpu_load_attestation(
            payload,
            config_sha256="1" * 64,
            implementation_sha256="2" * 64,
            expected_purpose="TRAINING_LAUNCH",
            current_runtime=_runtime(
                "GPU-00000000-0000-0000-0000-000000000099"
            ),
            require_fresh=True,
            require_live_parent=False,
        )


def test_production_train_and_finalize_api_require_attestation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(subject, "_require_torch", lambda: object())
    monkeypatch.setattr(subject, "_configure_production_determinism", lambda: None)
    monkeypatch.setattr(subject, "_validate_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "_verify_parent_protocol", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "_validate_runtime_for_config", lambda *_args, **_kwargs: (_runtime(), "r" * 64))
    monkeypatch.setattr(subject, "implementation_sha256", lambda: "2" * 64)
    monkeypatch.setattr(subject, "_canonical_sha256", lambda value: "1" * 64 if value is config else attestation.canonical_sha256(value))
    with pytest.raises(attestation.GpuLoadAttestationError, match="gpu-attestation"):
        subject.train_population(
            config,
            output_dir=tmp_path / "agents",
            family="mf",
            production=True,
        )

    monkeypatch.setattr(subject, "_namespace_finalize_lock", lambda *_args, **_kwargs: nullcontext())
    with pytest.raises(attestation.GpuLoadAttestationError, match="gpu-attestation"):
        subject.finalize_artifacts(
            config,
            output_dir=tmp_path / "agents",
            report_path=tmp_path / "report.json",
            agent_registry_path=tmp_path / "agents.csv",
            selection_ledger_path=tmp_path / "selection.csv",
            training_ledger_path=tmp_path / "training.parquet",
            trajectory_path=tmp_path / "trajectories.parquet",
            event_path=tmp_path / "events.parquet",
            production=True,
        )

    monkeypatch.setattr(
        subject, "_require_canonical_production_path", lambda *_args, **_kwargs: None
    )

    def forbidden_recovery(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("recovery ran before finalizer attestation validation")

    monkeypatch.setattr(
        subject, "_recover_finalization_atomic_temporaries", forbidden_recovery
    )
    with pytest.raises(attestation.GpuLoadAttestationError, match="gpu-attestation"):
        subject._finalize_artifacts_locked(
            config,
            output_dir=tmp_path / "agents",
            report_path=tmp_path / "report.json",
            agent_registry_path=tmp_path / "agents.csv",
            selection_ledger_path=tmp_path / "selection.csv",
            training_ledger_path=tmp_path / "training.parquet",
            trajectory_path=tmp_path / "trajectories.parquet",
            event_path=tmp_path / "events.parquet",
            production=True,
        )


def test_canonical_production_api_rejects_nmf_without_gate_and_family_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(subject, "_require_torch", lambda: object())
    monkeypatch.setattr(subject, "_configure_production_determinism", lambda: None)
    monkeypatch.setattr(subject, "_validate_config", lambda *_args, **_kwargs: None)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="NMF requires"):
        subject.train_population(
            config, output_dir=output, family="nmf", production=True
        )
    with pytest.raises(ValueError, match="family=all is forbidden"):
        subject.train_population(
            config, output_dir=output, family="all", production=True
        )
    assert not output.exists()


def test_nmf_wait_times_out_before_any_output_lock_attempt_or_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = run_dir / subject.SERIAL_RELEASE_FILENAME
    normalized_attestation = {
        "run_identity": {
            "transaction_id": "00000000-0000-0000-0000-000000000013",
            "run_dir": str(run_dir.resolve()),
        },
        "attestation_nonce": "00000000-0000-0000-0000-000000000014",
        "attestation_sha256": "3" * 64,
    }
    monkeypatch.setattr(subject, "_require_torch", lambda: object())
    monkeypatch.setattr(subject, "_configure_production_determinism", lambda: None)
    monkeypatch.setattr(subject, "_validate_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "_validate_runtime_for_config", lambda *_args, **_kwargs: ({}, "r" * 64))
    monkeypatch.setattr(subject, "_verify_parent_protocol", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "implementation_sha256", lambda: "2" * 64)
    monkeypatch.setattr(subject, "_validate_gpu_load_attestation", lambda *_args, **_kwargs: normalized_attestation)
    monkeypatch.setattr(subject, "_gpu_attestation_binding", lambda *_args, **_kwargs: {"bound": True})

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("output namespace or training was reached before release")

    monkeypatch.setattr(subject, "_assert_finalize_not_active", forbidden)
    monkeypatch.setattr(subject, "_train_or_resume_agent", forbidden)
    output = tmp_path / "output-must-not-exist"
    with pytest.raises(TimeoutError, match="serial-release"):
        subject.train_population(
            config,
            output_dir=output,
            family="nmf",
            production=True,
            gpu_attestation={"synthetic": True},
            serial_release_gate=gate,
            serial_release_nonce="00000000-0000-0000-0000-000000000015",
            serial_release_deadline_utc=(
                datetime.now(timezone.utc) + timedelta(milliseconds=30)
            ).isoformat(),
        )
    assert not output.exists()
    assert not list(tmp_path.rglob("*.lock"))
    assert not list(tmp_path.rglob("start.json"))


def test_serial_release_contract_covers_success_forgery_preexist_path_timeout_and_mf_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path.resolve()
    gate = run_dir / subject.SERIAL_RELEASE_FILENAME
    release_nonce = "00000000-0000-0000-0000-000000000015"
    gpu = {
        "run_identity": {
            "transaction_id": "00000000-0000-0000-0000-000000000013",
            "run_dir": str(run_dir),
        },
        "attestation_nonce": "00000000-0000-0000-0000-000000000014",
        "attestation_sha256": "3" * 64,
    }

    def deadline(seconds: float = 1.0) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def release(deadline_utc: str, **changes: object) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": subject.SERIAL_RELEASE_SCHEMA_VERSION,
            "task_id": subject.TASK_ID,
            "family": "nmf",
            "prerequisite_family": "mf",
            "prerequisite_exit_code": 0,
            "transaction_id": gpu["run_identity"]["transaction_id"],
            "run_dir": str(run_dir),
            "attestation_nonce": gpu["attestation_nonce"],
            "attestation_sha256": gpu["attestation_sha256"],
            "release_nonce": release_nonce,
            "waiter_pid": os.getpid(),
            "config_sha256": "1" * 64,
            "implementation_sha256": "2" * 64,
            "deadline_utc": deadline_utc,
            "mf_pid": 11,
            "mf_process_created_unix_ns": 12,
            "released_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        value.update(changes)
        return value

    def delayed_write(value: dict[str, object]) -> threading.Thread:
        def writer() -> None:
            time.sleep(0.02)
            value["released_at_utc"] = datetime.now(timezone.utc).isoformat()
            subject._atomic_json(value, gate)

        worker = threading.Thread(target=writer)
        worker.start()
        return worker

    success_deadline = deadline()
    worker = delayed_write(release(success_deadline))
    witness = subject._wait_for_nmf_serial_release(
        gate_path=gate,
        release_nonce=release_nonce,
        deadline_utc=success_deadline,
        config_sha256="1" * 64,
        implementation_sha256="2" * 64,
        gpu_attestation=gpu,
        poll_interval_seconds=0.005,
        emit_ready=False,
    )
    worker.join()
    assert witness["mf_pid"] == 11
    gate.unlink()

    preexisting_deadline = deadline()
    subject._atomic_json(release(preexisting_deadline), gate)
    with pytest.raises(ValueError, match="pre-existed"):
        subject._wait_for_nmf_serial_release(
            gate_path=gate,
            release_nonce=release_nonce,
            deadline_utc=preexisting_deadline,
            config_sha256="1" * 64,
            implementation_sha256="2" * 64,
            gpu_attestation=gpu,
            emit_ready=False,
        )
    gate.unlink()

    for changes in (
        {"release_nonce": "00000000-0000-0000-0000-000000000099"},
        {"prerequisite_exit_code": 7},
    ):
        forged_deadline = deadline()
        worker = delayed_write(release(forged_deadline, **changes))
        with pytest.raises(ValueError, match="identity or timeline"):
            subject._wait_for_nmf_serial_release(
                gate_path=gate,
                release_nonce=release_nonce,
                deadline_utc=forged_deadline,
                config_sha256="1" * 64,
                implementation_sha256="2" * 64,
                gpu_attestation=gpu,
                poll_interval_seconds=0.005,
                emit_ready=False,
            )
        worker.join()
        gate.unlink()

    with pytest.raises(ValueError, match="unique file"):
        subject._wait_for_nmf_serial_release(
            gate_path=run_dir / "arbitrary.json",
            release_nonce=release_nonce,
            deadline_utc=deadline(),
            config_sha256="1" * 64,
            implementation_sha256="2" * 64,
            gpu_attestation=gpu,
            emit_ready=False,
        )
    with pytest.raises(TimeoutError, match="serial-release"):
        subject._wait_for_nmf_serial_release(
            gate_path=gate,
            release_nonce=release_nonce,
            deadline_utc=deadline(0.03),
            config_sha256="1" * 64,
            implementation_sha256="2" * 64,
            gpu_attestation=gpu,
            poll_interval_seconds=0.005,
            emit_ready=False,
        )

    late_deadline = deadline(0.005)
    worker = delayed_write(release(late_deadline))
    with pytest.raises(TimeoutError, match="after the transaction deadline"):
        subject._wait_for_nmf_serial_release(
            gate_path=gate,
            release_nonce=release_nonce,
            deadline_utc=late_deadline,
            config_sha256="1" * 64,
            implementation_sha256="2" * 64,
            gpu_attestation=gpu,
            poll_interval_seconds=0.05,
            emit_ready=False,
        )
    worker.join()
    gate.unlink()

    original_validator = subject._validate_nmf_serial_release_payload

    def slow_validator(*args: object, **kwargs: object) -> dict[str, object]:
        time.sleep(0.12)
        return original_validator(*args, **kwargs)

    monkeypatch.setattr(subject, "_validate_nmf_serial_release_payload", slow_validator)
    validation_deadline = deadline(0.1)
    worker = delayed_write(release(validation_deadline))
    with pytest.raises(TimeoutError, match="validation crossed"):
        subject._wait_for_nmf_serial_release(
            gate_path=gate,
            release_nonce=release_nonce,
            deadline_utc=validation_deadline,
            config_sha256="1" * 64,
            implementation_sha256="2" * 64,
            gpu_attestation=gpu,
            poll_interval_seconds=0.005,
            emit_ready=False,
        )
    worker.join()
    gate.unlink()


def test_supervisor_contains_two_fresh_gates_job_binding_and_strict_crash_marker() -> None:
    script = (ROOT / "scripts/run_t9_1_3_production.ps1").read_text(encoding="utf-8")
    assert "[int]$TotalDeadlineHours = 36" in script
    assert script.count("New-GpuLoadAttestation -LoadGate") == 2
    assert "--gpu-attestation" in script
    assert "$mf = Start-JobBoundPythonChild -Role 'mf'" in script
    assert "$nmf = Start-JobBoundPythonChild -Role 'nmf'" in script
    assert "$finalize = Start-JobBoundPythonChild -Role 'finalize'" in script
    assert "Add-ProcessToKillOnCloseJob -Process $process -Role $Role" in script
    assert "job_object_bound_before_python_payload_release" in script
    assert "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE" in script
    assert "INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL" in script
    assert "FINALIZATION_FAILED_NO_VALID_SEAL" in script
    assert "Test-StrictInvalidatedFinalizationCrashMarker" in script
    assert "MANUAL_OPERATOR_RECOVERY_REQUIRED" in script
    assert "finalizeRecoveryOnly" not in script
    assert "$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'" in script
    assert "$env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = '0'" in script
    assert "--supervisor-nmf-release-gate" in script
    assert "Wait-NmfSerialReleaseReady" in script
    assert "Wait-MfWithBlockedNmf" in script
    assert "Wait-NmfSerialReleaseConsumed" in script
    assert "$ioDrainedAfterExit = $false" in script
    assert "Close-HiddenChildIo -Process $Process" in script
    assert "NMF_WAITER_EXITED_BEFORE_RELEASE" in script
    assert "Publish-NmfSerialRelease" in script
    assert script.index("Wait-NmfSerialReleaseReady -Process $nmf") < script.index(
        "Wait-MfWithBlockedNmf -MfProcess $mf"
    )
    assert script.index("Wait-MfWithBlockedNmf -MfProcess $mf") < script.index(
        "Publish-NmfSerialRelease -MfProcess $mf"
    )
    assert script.index("Publish-NmfSerialRelease -MfProcess $mf") < script.index(
        "Wait-NmfSerialReleaseConsumed -Process $nmf"
    )
    assert script.index("Wait-NmfSerialReleaseConsumed -Process $nmf") < script.index(
        "Wait-SingleChild -Role 'nmf' -Process $nmf"
    )
    assert "nmf_serial_release_witness = $nmfRelease" in script
    assert "nmf_serial_consumption_witness = $nmfConsumed" in script
    assert "nmf_serial_release_consumption_verified = $supervisorReleaseConsumedPass" in script
    assert "shared_training_attestation_sha256" in script
    assert "t9.1.3-production-supervisor-v3" in script
    assert "t9.1.3-production-serialized-training-outcome-v3" in script


def test_static_attestation_self_test_has_two_gates_and_no_gpu_query() -> None:
    result = attestation.synthetic_attestation_self_test()
    assert result["status"] == "PASS"
    assert result["training_gate_pass"] is True
    assert result["finalizer_gate_pass"] is True
    assert result["gpu_queried"] is False
    assert result["production_started"] is False
