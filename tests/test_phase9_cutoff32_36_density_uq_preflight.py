from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import (
    phase9_cutoff32_36_density_uq_preflight as subject,
)


ROOT = Path(__file__).resolve().parents[1]


def _live_identity(config):
    identity = {
        "task_id": subject.TASK_ID,
        "schema_version": subject.RUN_IDENTITY_SCHEMA,
        "run_id": "2e921c62-3d2e-43d8-a1bd-e62dc7069ddd",
        "config_analysis_sha256": subject._sha(config),
        "execution_bindings": subject._execution_bindings(ROOT, config),
        "claim_state": dict(subject.CLAIM_BOUNDARY),
        "created_utc": "2026-07-27T00:00:00+00:00",
    }
    identity["analysis_sha256"] = subject._sha(identity)
    return identity


def _worker_attestation(identity):
    return {
        "worker_pid": 1234,
        "confirmation_source_sha256": identity["execution_bindings"][
            "confirmation_source"
        ]["sha256"],
        "paired_cluster_uq_source_sha256": identity["execution_bindings"][
            "paired_cluster_uq_source"
        ]["sha256"],
    }


def _lock_contender(root_text, config, queue) -> None:
    try:
        with subject._owner_lock(Path(root_text), config):
            queue.put("acquired")
    except BaseException as exc:
        queue.put(type(exc).__name__)


def _hard_exit_with_lock(root_text, config, ready) -> None:
    with subject._owner_lock(Path(root_text), config):
        ready.set()
        os._exit(0)


def test_live_config_pins_parents_domain_and_claim_boundary() -> None:
    config, calibration, extension = subject.load_config(ROOT)
    cells = subject._cells(config)
    assert len(cells) == 72
    assert {cell.dimension for cell in cells} == {84, 96, 108}
    assert {cell.cluster_count for cell in cells} == {12, 384}
    assert config["claim_boundary"] == subject.CLAIM_BOUNDARY
    assert calibration["verdict"] == "NO_GO_PAIRED_CLUSTER_UQ_CALIBRATION"
    assert extension["verdict"] == "PASS_PAIRED_CLUSTER_UQ_POWER_EXTENSION"
    assert extension["selected_formal_clusters_per_state"] == 384


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("gates", "minimum_cell_coverage_wilson_lcb"), 0.89),
        (("families", "heavy_tail_rare_coherent", "rare_probability"), 0.2),
        (("margin",), 0.11),
    ],
)
def test_preregistered_gate_and_family_drift_is_rejected(
    tmp_path, monkeypatch, path, value
) -> None:
    config = json.loads((ROOT / subject.CONFIG_PATH).read_text(encoding="utf-8"))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    relative = "mutated_config.json"
    (tmp_path / relative).write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(subject, "CONFIG_PATH", relative)
    with pytest.raises(ValueError, match="identity/claim/domain|family/gate"):
        subject.load_config(tmp_path)


@pytest.mark.parametrize(
    "family_name",
    [
        "low_energy_balanced",
        "heavy_tail_rare_coherent",
        "heteroskedastic_coherent",
    ],
)
def test_generator_is_physical_and_hits_analytic_effect(
    family_name: str,
) -> None:
    config, _, _ = subject.load_config(ROOT)
    left, right, truth = subject._physical_density_trial(
        dimension=84,
        count=12,
        true_distance=0.12,
        family=config["families"][family_name],
        seed=1234,
    )
    assert truth == pytest.approx(0.12, abs=1e-12)
    for stack in (left, right):
        validated = subject._validate_density_stack(stack, family_name)
        assert validated.shape == (12, 84, 84)
        assert np.trace(validated, axis1=1, axis2=2) == pytest.approx(
            np.ones(12), abs=1e-9
        )


def test_physical_ucb_rejects_non_psd_or_non_trace_one_input() -> None:
    valid = np.repeat((np.eye(4) / 4)[None, :, :], 12, axis=0)
    invalid = valid.copy()
    invalid[0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="non-physical"):
        subject.paired_density_trace_ucb_physical(
            invalid,
            valid,
            confidence=0.95,
            multiplier_replicates=199,
            seed=1,
            calibration_factor=1.0,
        )


def test_wilson_bounds_are_not_point_rates() -> None:
    lower, upper = subject._wilson_bounds(167, 256)
    simultaneous_lower, simultaneous_upper = subject._wilson_bounds(
        167,
        256,
        confidence=0.95,
        comparisons=128,
    )
    assert 167 / 256 > 0.65
    assert lower < 0.65
    assert upper > 167 / 256
    assert simultaneous_lower < lower
    assert simultaneous_upper > upper


def test_global_wilson_contract_is_analytically_attainable() -> None:
    config, _, _ = subject.load_config(ROOT)
    feasibility = subject._wilson_feasibility(config)
    assert feasibility["global_comparisons"] == 128
    assert feasibility["all_successes_wilson_lcb"] >= 0.90
    assert feasibility["zero_successes_wilson_ucb"] <= 0.05
    assert feasibility["attainable"] is True


def _synthetic_records(config, *, local_successes: int):
    records = []
    split = config["confirmation_split"]
    for cell in subject._cells(config):
        for trial in range(int(split["trial_count_per_cell"])):
            if cell.true_distance == 0.0:
                bound = 0.05
            elif cell.true_distance == 0.05:
                bound = 0.09 if trial < local_successes else 0.11
            elif cell.true_distance == 0.1:
                bound = 0.11
            else:
                bound = 0.13
            records.append(
                {
                    "split": "confirmation",
                    "family": cell.family,
                    "dimension": cell.dimension,
                    "cluster_count": cell.cluster_count,
                    "true_distance": cell.true_distance,
                    "trial": trial,
                    "trial_seed": subject._seed(
                        int(split["trial_seed_base"]),
                        "confirmation",
                        cell.family,
                        cell.dimension,
                        cell.cluster_count,
                        cell.true_distance,
                        trial,
                    ),
                    "multiplier_seed": subject._seed(
                        int(split["multiplier_seed_base"]),
                        "confirmation",
                        cell.family,
                        cell.dimension,
                        cell.cluster_count,
                        cell.true_distance,
                        trial,
                    ),
                    "estimate": bound,
                    "raw_radius": 0.0,
                    "power_primary": bool(
                        config["families"][cell.family]["power_primary"]
                    ),
                }
            )
    return records


def test_point_pass_but_wilson_lcb_fail_is_no_go() -> None:
    config, calibration, extension = subject.load_config(ROOT)
    records = _synthetic_records(config, local_successes=167)
    report, _ = subject.build_report(
        ROOT,
        config,
        calibration,
        extension,
        _live_identity(config),
        {"passed": True, "analysis_sha256": "fixture-preflight"},
        records,
        [{"cell": asdict(cell)} for cell in subject._cells(config)],
    )
    assert report["coverage_summary"]["formal_count_all_passed"] is True
    assert report["confirmation_power_passed"] is False
    assert report["selected_formal_clusters_per_state"] is None
    assert report["verdict"] == subject.NO_GO_VERDICT
    local = next(row for row in report["power_ledger"] if row["true_distance"] == 0.05)
    assert local["global_iut_pass"] is False


def test_owner_lock_rejects_concurrent_supervisor(tmp_path) -> None:
    config = {"artifact_paths": {"owner_lock": "run/owner.lock"}}
    with subject._owner_lock(tmp_path, config):
        with pytest.raises(RuntimeError, match="already exists"):
            with subject._owner_lock(tmp_path, config):
                pass


def test_owner_lock_rejects_spawned_competitor(tmp_path) -> None:
    config = {"artifact_paths": {"owner_lock": "run/owner.lock"}}
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with subject._owner_lock(tmp_path, config):
        process = context.Process(
            target=_lock_contender,
            args=(str(tmp_path), config, queue),
        )
        process.start()
        process.join(timeout=15)
        assert process.exitcode == 0
        assert queue.get(timeout=5) == "RuntimeError"


def test_owner_lock_cleans_after_exception(tmp_path) -> None:
    config = {"artifact_paths": {"owner_lock": "run/owner.lock"}}
    lock_path = tmp_path / "run" / "owner.lock"
    with pytest.raises(RuntimeError, match="injected"):
        with subject._owner_lock(tmp_path, config):
            raise RuntimeError("injected")
    assert not lock_path.exists()


def test_owner_lock_disappearance_is_terminal(tmp_path) -> None:
    config = {"artifact_paths": {"owner_lock": "run/owner.lock"}}
    lock_path = tmp_path / "run" / "owner.lock"
    with pytest.raises(RuntimeError, match="disappeared"):
        with subject._owner_lock(tmp_path, config):
            lock_path.unlink()


def test_hard_kill_leaves_stale_lock_for_read_only_diagnosis_and_manual_archive(
    tmp_path,
) -> None:
    config = {"artifact_paths": {"owner_lock": "run/owner.lock"}}
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_hard_exit_with_lock,
        args=(str(tmp_path), config, ready),
    )
    process.start()
    assert ready.wait(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
    lock_path = tmp_path / "run" / "owner.lock"
    stale = json.loads(lock_path.read_text(encoding="utf-8"))
    subject._verify_self_hash(stale, "stale owner lock")
    with pytest.raises(RuntimeError, match="already exists"):
        with subject._owner_lock(tmp_path, config):
            pass
    archive = lock_path.with_suffix(".stale.archived")
    lock_path.replace(archive)
    with subject._owner_lock(tmp_path, config):
        assert lock_path.exists()
    assert archive.exists()


def test_chunk_rejects_power_primary_flag_tamper() -> None:
    config, _, _ = subject.load_config(ROOT)
    run_identity = _live_identity(config)
    cell = subject._cells(config)[0]
    record = {
        "split": "confirmation",
        "family": cell.family,
        "dimension": cell.dimension,
        "cluster_count": cell.cluster_count,
        "true_distance": cell.true_distance,
        "trial": 0,
        "trial_seed": subject._seed(
            config["confirmation_split"]["trial_seed_base"],
            "confirmation",
            cell.family,
            cell.dimension,
            cell.cluster_count,
            cell.true_distance,
            0,
        ),
        "multiplier_seed": subject._seed(
            config["confirmation_split"]["multiplier_seed_base"],
            "confirmation",
            cell.family,
            cell.dimension,
            cell.cluster_count,
            cell.true_distance,
            0,
        ),
        "estimate": 0.0,
        "raw_radius": 0.1,
        "power_primary": not bool(config["families"][cell.family]["power_primary"]),
    }
    reduced = json.loads(json.dumps(config))
    reduced["confirmation_split"]["trial_count_per_cell"] = 1
    reduced_identity = _live_identity(reduced)
    chunk = subject._chunk_payload(
        reduced,
        cell,
        [record],
        reduced_identity,
        _worker_attestation(reduced_identity),
    )
    with pytest.raises(ValueError, match="record drift"):
        subject._validate_chunk(ROOT, reduced, cell, chunk, reduced_identity)


def test_run_identity_rejects_live_binding_tamper(tmp_path, monkeypatch) -> None:
    config = {
        "artifact_paths": {"run_identity": "run/run_identity.json"},
        "claim_boundary": dict(subject.CLAIM_BOUNDARY),
    }
    snapshot = {"config": {"path": "config.json", "bytes": 3, "sha256": "a" * 64}}
    monkeypatch.setattr(subject, "_execution_bindings", lambda root, live: snapshot)
    identity = subject._load_or_create_run_identity(tmp_path, config)
    snapshot = {"config": {"path": "config.json", "bytes": 4, "sha256": "b" * 64}}
    monkeypatch.setattr(subject, "_execution_bindings", lambda root, live: snapshot)
    with pytest.raises(RuntimeError, match="execution identity"):
        subject._verify_execution_identity_live(tmp_path, config, identity)


def test_chunk_rejects_run_identity_tamper() -> None:
    config, _, _ = subject.load_config(ROOT)
    reduced = json.loads(json.dumps(config))
    reduced["confirmation_split"]["trial_count_per_cell"] = 1
    identity = _live_identity(reduced)
    cell = subject._cells(reduced)[0]
    record = _synthetic_records(reduced, local_successes=1)[0]
    chunk = subject._chunk_payload(
        reduced,
        cell,
        [record],
        identity,
        _worker_attestation(identity),
    )
    tampered = json.loads(json.dumps(identity))
    tampered["run_id"] = "65b0ca65-d96e-46f8-b755-cf72be1f83f0"
    tampered["analysis_sha256"] = subject._sha(
        {key: value for key, value in tampered.items() if key != "analysis_sha256"}
    )
    with pytest.raises(ValueError, match="identity/accounting"):
        subject._validate_chunk(ROOT, reduced, cell, chunk, tampered)


def test_main_returns_nonzero_for_no_go(monkeypatch) -> None:
    monkeypatch.setattr(
        subject,
        "write_artifacts",
        lambda: {
            "analysis_sha256": "x",
            "verdict": subject.NO_GO_VERDICT,
            "selected_formal_clusters_per_state": None,
            "pilot_domain_factor_coverage_calibrated": False,
        },
    )
    assert subject.main([]) == 2
