from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.phase9_immutable_object_store import ImmutableObjectStore
from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    T04CellSpec,
    build_cell_plan,
)
from cnn_fpga.benchmark.phase9_powered_twin_preflight import (
    FAIL_VERDICT,
    ResourcePreflightFailure,
    ResourcePreflightSupervisor,
    _audit_resource_seed_ledger,
    _receipt_metrics,
    _resource_seed_addresses,
    _sha,
    _immutable_json,
    _validate_npy_payload,
    assert_seed_firewall,
    isolated_preflight_paths,
    no_copy_inventory,
    profile_cells,
    resource_gate_decision,
    stratified_projection,
    streaming_statistics_dry_run,
    validate_continuous_sampling,
    validate_statistics_profile,
)
from cnn_fpga.benchmark.phase9_powered_twin_runtime import (
    ActiveOwnerError,
    OwnerLease,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "phase9"
    / "t_risk_20260728_04_powered_twin_qualification.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sampling(
    *,
    count: int = 3,
    active: int = 2,
    overlap: int = 4,
) -> dict:
    return {
        "sample_count": count,
        "active_child_sample_count": active,
        "maximum_observed_live_children": overlap,
        "first_sample": {"monotonic_seconds": 0.0},
        "last_sample": {"monotonic_seconds": 1.0},
    }


def test_resource_seed_and_artifact_namespaces_never_touch_formal(
    tmp_path: Path,
) -> None:
    config = _config()
    evidence = assert_seed_firewall(config)
    assert evidence["formal_seed_addresses_accessed"] is False
    assert evidence["seed_namespace_pass"] is True
    resource = evidence["resource_interval_half_open"]
    for formal in evidence["formal_intervals_half_open"].values():
        assert max(resource[0], formal[0]) >= min(resource[1], formal[1])

    preflight_root, paths = isolated_preflight_paths(
        tmp_path, config, run_id="fixture1"
    )
    assert preflight_root == tmp_path / "runs" / "t04_resource_preflight_fixture1"
    for value in paths.values():
        preflight = (tmp_path / value).resolve()
        for key in ("object_store", "staging_directory", "receipt_directory"):
            formal = (tmp_path / config["artifact_paths"][key]).resolve()
            assert preflight != formal
            assert preflight not in formal.parents
            assert formal not in preflight.parents

    overlap = copy.deepcopy(config)
    overlap["artifact_paths"]["object_store"] = paths["object_store"]
    with pytest.raises(RuntimeError, match="overlaps"):
        isolated_preflight_paths(tmp_path, overlap, run_id="fixture1")


def test_seed_full_address_range_and_overlap_mutations_fail() -> None:
    config = _config()
    evidence = assert_seed_firewall(config)
    assert max(evidence["maximum_resource_addresses"].values()) < evidence[
        "resource_interval_half_open"
    ][1]

    too_small = copy.deepcopy(config)
    too_small["seed_registry"]["resource_preflight"]["count"] = 1_000_001
    with pytest.raises(RuntimeError, match="full preflight seed"):
        assert_seed_firewall(too_small)

    overlapping = copy.deepcopy(config)
    overlapping["seed_registry"]["resource_preflight"]["start"] = overlapping[
        "seed_registry"
    ]["physical"]["start"]
    with pytest.raises(RuntimeError, match="overlaps formal"):
        assert_seed_firewall(overlapping)


def test_frozen_full_profile_matches_formal_lpt_and_representatives() -> None:
    config = _config()
    cells = build_cell_plan(config)
    formal_peak, representatives = profile_cells(config, cells)
    assert [cell.plan_index for cell in formal_peak] == [478, 480, 482, 484]
    assert [cell.sample_count for cell in formal_peak] == [4608] * 4
    assert [cell.layer for cell in formal_peak] == ["fault"] * 4
    assert [cell.backend for cell in formal_peak] == ["A"] * 4
    assert [cell.plan_index for cell in representatives] == [
        388, 389, 403, 507
    ]
    assert [cell.layer for cell in representatives] == [
        "shared",
        "shared",
        "logical",
        "probe",
    ]
    mutated = copy.deepcopy(config)
    mutated["resource_contract"]["profile_plan"]["formal_lpt_four_worker_peak"][
        "full_frozen_denominator"
    ] = False
    with pytest.raises(RuntimeError, match="full frozen denominator"):
        profile_cells(mutated, cells)


def _write_seed_ledger(
    path: Path,
    config: dict,
    cell: T04CellSpec,
    *,
    mutation: str | None = None,
) -> tuple[str, ...]:
    from cnn_fpga.benchmark.phase9_fresh_twin_qualification import (
        LEDGER_FIELDS,
    )
    from cnn_fpga.benchmark.phase9_powered_twin_qualification import (
        EXTRA_FIELDS,
    )
    from cnn_fpga.benchmark.phase9_powered_twin_contract import (
        cluster_root_id,
    )

    header = tuple(LEDGER_FIELDS) + tuple(EXTRA_FIELDS)
    rows: list[dict[str, object]] = []
    for row_index in range(cell.expected_rows):
        position = row_index // cell.horizon
        round_index = row_index % cell.horizon
        physical, heldout = _resource_seed_addresses(
            config,
            cell,
            position=position,
            round_index=round_index,
        )
        row: dict[str, object] = {field: "" for field in header}
        trajectory_id = (
            f"{cell.cell_base}|c{cell.cutoff}|{cell.backend}|p{position:04d}"
            if cell.layer == "fault"
            else ""
        )
        row_id = (
            f"{trajectory_id}|r{round_index:03d}"
            if trajectory_id
            else (
                f"{cell.layer}|c{cell.cutoff}|{cell.cell_base}|"
                f"{cell.backend}|p{position:04d}"
            )
        )
        if cell.layer == "shared":
            cell_id = (
                f"ab/c{cell.cutoff}/shared/{cell.initial_state}/{cell.action}"
            )
        elif cell.layer == "logical":
            cell_id = (
                f"ab/c{cell.cutoff}/logical/{cell.logical_label}/{cell.action}"
            )
        elif cell.layer == "probe":
            cell_id = f"ab/c{cell.cutoff}/probe/{cell.probe_id}"
        else:
            cell_id = f"ab/c{cell.cutoff}/fault/{cell.scenario}"
        row.update(
            {
                "row_id": row_id,
                "row_schema": "PHASE9-POWERED-TWIN-ROUND-LEDGER-V1",
                "layer": cell.layer,
                "cell_base": cell.cell_base,
                "cell_id": cell_id,
                "backend": cell.backend,
                "backend_id": (
                    "PHASE9-BACKEND-A-JOINT-FOCK-QUTRIT-GKSL-V1"
                    if cell.backend == "A"
                    else "PHASE9-BACKEND-B-DENSE-STRANG-ANALYTIC-KRAUS-V1"
                ),
                "cutoff": cell.cutoff,
                "convergence_role": cell.convergence_role,
                "seed": physical,
                "seed_position": position,
                "trajectory_id": trajectory_id,
                "round_index": round_index,
                "terminal_round": str(
                    round_index == cell.horizon - 1
                ),
                "action": cell.action,
                "probe_id": cell.probe_id,
                "scenario": cell.scenario,
                "initial_state": cell.initial_state,
                "logical_label": cell.logical_label,
                "rng_namespace": (
                    "NUMPY_SEEDSEQUENCE_ADDRESSED"
                    if cell.backend == "A"
                    else "BLAKE2B_ADDRESS_PYTHON_RANDOM_BOX_MULLER"
                ),
                "archive_chunk": cell.chunk_id,
                "archive_row_index": row_index,
                "density_index": (
                    position
                    if cell.density_retention != "none"
                    and (
                        cell.layer != "fault"
                        or round_index == cell.horizon - 1
                    )
                    else -1
                ),
                "raw_iq_index": row_index,
                "heldout_iq_index": row_index,
                "conservation_pass": "True",
                "cluster_root_id": cluster_root_id(
                    config, cell, position
                ),
                "physical_seed_address": physical,
                "heldout_seed_address": heldout,
                "fault_state_index": (
                    position
                    // int(
                        config["formal_matrix"]["fault_clusters_per_state"]
                    )
                    if cell.layer == "fault"
                    else ""
                ),
                "fault_within_state_index": (
                    position
                    % int(
                        config["formal_matrix"]["fault_clusters_per_state"]
                    )
                    if cell.layer == "fault"
                    else ""
                ),
            }
        )
        rows.append(row)
    if mutation == "formal_physical":
        formal = int(config["seed_registry"]["physical"]["start"])
        rows[0]["seed"] = formal
        rows[0]["physical_seed_address"] = formal
    elif mutation == "wrong_resource_physical":
        rows[0]["seed"] = int(rows[0]["seed"]) + 1
        rows[0]["physical_seed_address"] = (
            int(rows[0]["physical_seed_address"]) + 1
        )
    elif mutation == "swapped_heldout":
        rows[0]["heldout_seed_address"], rows[1][
            "heldout_seed_address"
        ] = (
            rows[1]["heldout_seed_address"],
            rows[0]["heldout_seed_address"],
        )
    elif mutation == "duplicate_row_id":
        rows[1]["row_id"] = rows[0]["row_id"]
    elif mutation == "unique_but_wrong_row_id":
        rows[0]["row_id"] = "coordinated-but-unique-row-id"
    elif mutation == "wrong_cell_id":
        rows[0]["cell_id"] = "ab/c44/shared/forged"
    elif mutation == "short_denominator":
        rows.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=header, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return header


@pytest.mark.parametrize(
    "mutation",
    (
        "formal_physical",
        "wrong_resource_physical",
        "swapped_heldout",
        "duplicate_row_id",
        "unique_but_wrong_row_id",
        "wrong_cell_id",
        "short_denominator",
    ),
)
def test_live_seed_ledger_semantics_reject_coordinated_rehash_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = _config()
    original = build_cell_plan(config)[389]
    cell = replace(original, sample_count=2, expected_rows=2)
    path = tmp_path / "ledger.csv"
    header = _write_seed_ledger(path, config, cell)
    evidence = _audit_resource_seed_ledger(
        path,
        config=config,
        cell=cell,
        expected_header=header,
    )
    assert evidence["observed_rows"] == 2
    assert evidence["formal_seed_addresses_accessed"] is False
    _write_seed_ledger(
        path,
        config,
        cell,
        mutation=mutation,
    )
    with pytest.raises(RuntimeError, match="resource ledger"):
        _audit_resource_seed_ledger(
            path,
            config=config,
            cell=cell,
            expected_header=header,
        )


def test_endpoint_only_and_insufficient_concurrency_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="endpoint-only"):
        validate_continuous_sampling(_sampling(count=2, active=2))
    with pytest.raises(RuntimeError, match="endpoint-only"):
        validate_continuous_sampling(_sampling(count=3, active=1))
    with pytest.raises(RuntimeError, match="concurrency"):
        validate_continuous_sampling(_sampling(overlap=3))


def test_npy_payload_rejects_trailing_bytes_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.npy"
    np.save(path, np.ones((2, 2), dtype="<f8"), allow_pickle=False)
    binding = {
        "path": path.relative_to(tmp_path).as_posix(),
        "role": "heldout_iq_npy",
    }
    _validate_npy_payload(
        tmp_path,
        binding,
        shape=(2, 2),
        dtype="<f8",
    )
    with path.open("ab") as handle:
        handle.write(b"coordinated-trailer")
    with pytest.raises(RuntimeError, match="trailing"):
        _validate_npy_payload(
            tmp_path,
            binding,
            shape=(2, 2),
            dtype="<f8",
        )

    np.save(
        path,
        np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype="<f8"),
        allow_pickle=False,
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        _validate_npy_payload(
            tmp_path,
            binding,
            shape=(2, 2),
            dtype="<f8",
        )
    validate_continuous_sampling(_sampling())


def test_duplicate_live_owner_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "owner.lock"
    first = OwnerLease(
        path,
        run_id="resource-fixture",
        config_sha256="1" * 64,
        plan_sha256="2" * 64,
    )
    second = OwnerLease(
        path,
        run_id="resource-fixture",
        config_sha256="1" * 64,
        plan_sha256="2" * 64,
    )
    first.acquire()
    try:
        with pytest.raises(ActiveOwnerError):
            second.acquire()
    finally:
        first.release()


def test_mock_worker_failure_records_attempt_and_cannot_pollute_formal(
    tmp_path: Path,
) -> None:
    config = _config()
    calls: list[list[dict]] = []

    def fail_group(worker_kwargs, **kwargs):
        calls.append(list(worker_kwargs))
        assert kwargs["sample_callback"] is not None
        for item in worker_kwargs:
            assert item["seed_namespace"] == "resource_preflight"
            assert item["sample_count_override"] is None
            assert item["artifact_paths_override"] is not None
        raise RuntimeError("fixture worker crash")

    supervisor = ResourcePreflightSupervisor(
        root=tmp_path,
        config=config,
        config_sha256="1" * 64,
        plan_sha256=config["plan_contract"]["canonical_plan_sha256"],
        run_id="worker_failure",
        source_snapshot_sha256="3" * 64,
        sample_interval_seconds=5.0,
        heartbeat_period_seconds=0.01,
        process_group_runner=fail_group,
        lineage_validator=lambda *args: {"passed": True},
    )
    with pytest.raises(ResourcePreflightFailure) as raised:
        supervisor.run()
    report = raised.value.report
    assert report["verdict"] == FAIL_VERDICT
    assert report["formal_seed_addresses_accessed"] is False
    assert report["formal_artifact_namespace_accessed"] is False
    assert all(value is None for value in report["claim_boundary"].values())
    completed = report["completed_stage_evidence"]
    assert completed["lineage_validation"] == {"passed": True}
    assert completed["seed_firewall"] is not None
    assert completed["profile_measurements"] == []
    assert completed["projection"] is None
    assert completed["resource_gate_decision"] is None
    assert len(calls) == 1

    preflight = tmp_path / "runs" / "t04_resource_preflight_worker_failure"
    attempts = [
        json.loads(line)
        for line in (preflight / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in attempts] == [
        "START_RESOURCE_PREFLIGHT",
        "FAIL_RESOURCE_PREFLIGHT",
    ]
    assert (preflight / "resource_preflight_failed.json").is_file()
    immutable_events = sorted((preflight / "attempt_events").glob("*.json"))
    assert len(immutable_events) == 2
    assert [json.loads(path.read_text())["event"] for path in immutable_events] == [
        "START_RESOURCE_PREFLIGHT",
        "FAIL_RESOURCE_PREFLIGHT",
    ]
    for key in ("object_store", "staging_directory", "receipt_directory"):
        assert not (tmp_path / config["artifact_paths"][key]).exists()
    with pytest.raises(RuntimeError, match="fresh run_id"):
        supervisor.run()


def test_late_wall_gate_failure_preserves_completed_stage_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cnn_fpga.benchmark import phase9_powered_twin_preflight as preflight

    config = _config()
    group_calls = 0

    def complete_group(worker_kwargs, **kwargs):
        nonlocal group_calls
        group_calls += 1
        assert kwargs["sample_callback"] is not None
        return [
            {
                "plan_index": int(item["cell"].plan_index),
                "profile_wall_seconds": 1.0,
            }
            for item in worker_kwargs
        ]

    stats = {
        "wall_seconds": 2.0,
        "peak_analysis_scratch_bytes": 17,
        "retained_density_physicality_profile": {
            "projected_full_serial_wall_seconds": 3.0,
        },
    }
    raw_seed_audit = {
        "formal_seed_addresses_accessed": False,
        "receipt_count": 8,
    }
    inventory = {
        "raw_status": "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT",
        "receipt_count": 8,
        "totals": {
            "object_bytes_unique": 123,
            "expected_rows": 227_328,
            "observed_rows": 227_328,
            "exception_rows": 0,
            "missing_rows": 0,
            "conservation_failures": 0,
        },
        "monolithic_archive": None,
        "merged_full_csv": None,
    }
    inventory_evidence = {"finalize_wall_seconds": 4.0}
    projection = {
        "cell_projections": [
            {"projected_transient_bytes": value}
            for value in (11, 7, 5, 3)
        ],
        "projected_formal_artifact_bytes": 456,
        "projected_formal_wall_seconds_at_frozen_concurrency": (
            float(config["resource_contract"]["maximum_wall_seconds"]) + 1.0
        ),
    }
    decision = {
        "checks": {
            "rss": True,
            "artifact": True,
            "disk": True,
            "wall": False,
            "inventory": True,
        },
        "passed": False,
        "decision_sha256": "d" * 64,
    }

    monkeypatch.setattr(
        preflight,
        "audit_resource_profile_receipts",
        lambda *args, **kwargs: raw_seed_audit,
    )
    monkeypatch.setattr(
        preflight,
        "no_copy_inventory",
        lambda *args, **kwargs: (inventory, inventory_evidence),
    )
    monkeypatch.setattr(
        preflight,
        "validate_statistics_profile",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        preflight,
        "validate_continuous_sampling",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        preflight,
        "stratified_projection",
        lambda *args, **kwargs: projection,
    )
    monkeypatch.setattr(
        preflight,
        "resource_gate_decision",
        lambda *args, **kwargs: decision,
    )

    supervisor = ResourcePreflightSupervisor(
        root=tmp_path,
        config=config,
        config_sha256="1" * 64,
        plan_sha256=config["plan_contract"]["canonical_plan_sha256"],
        run_id="late_wall_failure",
        source_snapshot_sha256="3" * 64,
        sample_interval_seconds=5.0,
        heartbeat_period_seconds=0.01,
        process_group_runner=complete_group,
        stats_runner=lambda *args, **kwargs: dict(stats),
        lineage_validator=lambda *args: {"passed": True},
    )
    with pytest.raises(ResourcePreflightFailure) as raised:
        supervisor.run()

    assert group_calls == 2
    report = raised.value.report
    assert report["error"] == "resource gates failed: wall"
    completed = report["completed_stage_evidence"]
    assert len(completed["profile_measurements"]) == 8
    assert completed["streaming_statistics_dry_run"]["wall_seconds"] == 2.0
    assert completed["raw_seed_audit"] == raw_seed_audit
    assert completed["inventory"] == inventory
    assert completed["inventory_no_copy_evidence"] == inventory_evidence
    assert completed["projection"] == projection
    assert completed["resource_gate_decision"] == decision
    assert completed["maximum_inflight_temp_bytes"] == 26
    assert completed["analysis_scratch_bytes"] == 17
    assert all(value is None for value in report["claim_boundary"].values())
    persisted = json.loads(
        (
            tmp_path
            / "runs"
            / "t04_resource_preflight_late_wall_failure"
            / "resource_preflight_failed.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["completed_stage_evidence"] == completed


def _tiny_cells() -> list[T04CellSpec]:
    base = T04CellSpec(
        plan_index=0,
        chunk_id="fixture_0",
        layer="shared",
        cutoff=4,
        backend="A",
        cell_base="fixture",
        pair_group_id="fixture",
        pair_group_index=0,
        sample_count=1,
        horizon=1,
        expected_rows=1,
        action="IDLE",
        density_retention="none",
        reset_estimand_scope="none",
    )
    return [
        replace(base, plan_index=index, chunk_id=f"fixture_{index}")
        for index in range(5)
    ]


def _tiny_store(tmp_path: Path) -> tuple[ImmutableObjectStore, list[T04CellSpec]]:
    store = ImmutableObjectStore(
        repository_root=tmp_path,
        object_root=tmp_path / "objects" / "sha256",
        staging_root=tmp_path / "staging",
        receipt_root=tmp_path / "receipts",
        task_id="T-RISK-20260728-04",
        run_id="inventory-fixture",
        config_sha256="1" * 64,
        plan_sha256="2" * 64,
        source_snapshot_sha256="3" * 64,
        seed_namespace="resource_preflight",
        runner_id="PHASE9-POWERED-TWIN-RAW-RUNNER-V1",
    )
    cells = _tiny_cells()
    for cell in cells:
        binding = store.put_bytes(
            f"payload-{cell.plan_index}".encode(),
            role="fixture_payload",
        )
        store.commit_receipt(
            cell=asdict(cell),
            objects=[binding],
            expected_rows=1,
            observed_rows=1,
            exception_rows=0,
            missing_rows=0,
            conservation_failures=0,
            source_snapshot_sha256="3" * 64,
            runtime_fingerprint={
                "runner_id": "PHASE9-POWERED-TWIN-RAW-RUNNER-V1",
                "python": [3, 12, 7],
                "numpy": "1.26.4",
                "scipy": "1.13.1",
                "psutil": "5.9.0",
                "platform": "test-platform",
                "thread_environment": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "seed_namespace": "resource_preflight",
            },
            reset_rows=0,
            reset_sidecar_rows=0,
        )
    return store, cells


def test_inventory_revalidates_content_without_zip_or_raw_copy(
    tmp_path: Path,
) -> None:
    store, cells = _tiny_store(tmp_path)
    before = {
        path.relative_to(store.object_root): path.stat().st_size
        for path in store.object_root.rglob("*")
        if path.is_file()
    }
    inventory, evidence = no_copy_inventory(store, cells)
    after = {
        path.relative_to(store.object_root): path.stat().st_size
        for path in store.object_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert inventory["receipt_count"] == 5
    assert inventory["monolithic_archive"] is None
    assert inventory["merged_full_csv"] is None
    assert evidence["raw_payload_bytes_copied_during_finalize"] == 0

    forbidden = store.object_root / "raw.zip"
    forbidden.write_bytes(b"not an archive, but a forbidden container name")
    with pytest.raises(RuntimeError, match="monolithic"):
        no_copy_inventory(store, cells)


def test_resource_byte_projection_only_deduplicates_explicit_alias() -> None:
    def binding(role: str, digest: str, size: int, path: str) -> dict:
        return {
            "role": role,
            "sha256": digest,
            "bytes": size,
            "path": path,
        }

    receipt = {
        "cell": {
            "chunk_id": "fixture",
            "plan_index": 389,
            "layer": "shared",
        },
        "diagnostics": {"expected_rows": 2, "reset_rows": 2},
        "receipt_sha256": "f" * 64,
        "objects": [
            binding("round_ledger_csv", "1" * 64, 50, "objects/1"),
            binding("primary_density_npy", "2" * 64, 100, "objects/2"),
            binding("rb_expected_density_npy", "2" * 64, 100, "objects/2"),
            binding(
                "rb_conditional_success_density_npy",
                "3" * 64,
                200,
                "objects/3",
            ),
            binding(
                "rb_sampled_stress_density_npy",
                "3" * 64,
                200,
                "objects/3",
            ),
        ],
    }
    metrics = _receipt_metrics(
        {
            "receipt": receipt,
            "pid": 1,
            "wall_seconds": 1.0,
        }
    )
    assert metrics["object_bytes_unique"] == 350
    assert metrics["explicit_alias_bytes"] == 100
    assert metrics["conservative_payload_bytes"] == 550
    receipt["cell"]["layer"] = "fault"
    fault_metrics = _receipt_metrics(
        {
            "receipt": receipt,
            "pid": 1,
            "wall_seconds": 1.0,
        }
    )
    assert fault_metrics["explicit_alias_bytes"] == 0
    assert fault_metrics["conservative_payload_bytes"] == 650


def test_immutable_json_is_atomic_fail_if_exists_under_race(
    tmp_path: Path,
) -> None:
    target = tmp_path / "immutable.json"
    values = [{"writer": 0}, {"writer": 1}]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_immutable_json, target, value)
            for value in values
        ]
    failures = [future.exception() for future in futures if future.exception()]
    assert len(failures) == 1
    assert "conflicting immutable" in str(failures[0])
    winner = json.loads(target.read_text(encoding="utf-8"))
    assert winner in values
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match="conflicting immutable"):
        _immutable_json(target, {"writer": 2})
    assert target.read_bytes() == before
    _immutable_json(target, winner)


def test_streaming_statistics_dry_run_has_full_shape_and_resource_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    config = _config()
    group_sizes: list[int] = []
    callback_dual_buffer_live: list[bool] = []
    callback_physicality_live: list[bool] = []
    eigvalsh_shapes: list[tuple[int, ...]] = []
    original_eigvalsh = np.linalg.eigvalsh

    def fast_signs(*, seed, replicates, cluster_root_ids):
        group_sizes.append(len(cluster_root_ids))
        assert all(
            root.startswith("resource/") for root in cluster_root_ids
        )
        return np.ones((replicates, len(cluster_root_ids)), dtype=np.int8)

    def sample() -> None:
        caller = inspect.currentframe().f_back
        local = caller.f_locals
        if "fixture" in local and "eigenvalues" in local:
            callback_physicality_live.append(True)
            assert local["fixture"].shape == (256, 132, 132)
            assert local["fixture"].dtype == np.dtype(np.complex64)
            assert local["stack"].shape == (8, 132, 132)
            assert local["stack"].dtype == np.dtype(np.complex128)
            assert local["hermitian"].shape == (8, 132, 132)
            assert local["eigenvalues"].shape == (8, 132)
            return
        dual_live = "source_left" in local and "source_right" in local
        callback_dual_buffer_live.append(dual_live)
        if dual_live:
            assert local["density_perturbation"].shape == (199, 132, 132)
            assert local["update"].shape == (199, 132, 132)
            assert local["source_left"].shape == (32, 132, 132)
            assert local["source_right"].shape == (32, 132, 132)
            assert local["block"].shape == (32, 132, 132)
            assert local["centered"].shape == (32, 132 * 132)

    def counted_eigvalsh(value):
        eigvalsh_shapes.append(tuple(value.shape))
        return original_eigvalsh(value)

    monkeypatch.setattr(np.linalg, "eigvalsh", counted_eigvalsh)
    result = streaming_statistics_dry_run(
        config,
        sample_callback=sample,
        sign_matrix_factory=fast_signs,
    )
    resource = config["seed_registry"]["resource_preflight"]
    assert result["gate_count"] == 3037
    assert result["replicates"] == 199
    assert result["streaming"] is True
    assert result["maximum_coexisting_gate_buffers"] == 1
    assert result["production_rademacher_generator_exercised"] is False
    assert result["conservative_dual_leg_max_exercised"] is True
    assert result["dual_leg_evaluation_count"] == 6074
    assert result["l1_accumulation_exercised"] is True
    assert result["largest_density_kernel_exercised"] is True
    assert result["largest_density_root_count"] == 4608
    assert result["largest_density_block_rows"] == 32
    assert result["largest_density_block_count"] == 144
    assert result["largest_density_source_buffer_count"] == 2
    assert result["largest_density_rss_callback_count"] == 144
    assert result["largest_density_perturbation_shape"] == [199, 132, 132]
    assert result["largest_density_perturbation_bytes"] == (
        199 * 132 * 132 * 16
    )
    assert result["largest_density_update_bytes"] == (
        result["largest_density_perturbation_bytes"]
    )
    assert result["largest_density_trace_norm_evaluations"] == 199
    assert eigvalsh_shapes.count((8, 132, 132)) == 96
    assert eigvalsh_shapes.count((132, 132)) == 199
    assert len(eigvalsh_shapes) == 295
    assert callback_physicality_live == [True] * 96
    assert callback_dual_buffer_live == [True] * 144 + [False]
    assert len(result["largest_density_kernel_sha256"]) == 64
    assert result["persistent_working_set_bytes"] == sum(
        result["persistent_working_set_components"].values()
    )
    assert result["largest_density_peak_live_bytes"] == sum(
        result["largest_density_peak_live_components"].values()
    )
    assert result["peak_explicit_working_set_bytes"] == (
        result["persistent_working_set_bytes"]
        + result["largest_density_peak_live_bytes"]
    )
    assert result["largest_density_peak_live_components"][
        "perturbation"
    ] == result["largest_density_perturbation_bytes"]
    assert result["largest_density_peak_live_components"]["update"] == (
        result["largest_density_update_bytes"]
    )
    physicality = result["retained_density_physicality_profile"]
    assert physicality["matrix_dimension"] == 132
    assert physicality["block_size"] == 8
    assert physicality["fixture_matrix_count"] == 256
    assert physicality["timed_repeats"] == 3
    assert physicality["timed_matrix_evaluations"] == 768
    assert physicality["rss_callback_count"] == 96
    assert physicality["projected_full_retained_count"] == 482_304
    assert physicality["projected_full_serial_wall_seconds"] > 0.0
    assert physicality["full_fixture_generated"] is False
    assert physicality["scientific_data_used"] is False
    assert physicality["scientific_verdict"] is None
    assert result["peak_analysis_scratch_bytes"] == max(
        result["peak_explicit_working_set_bytes"],
        physicality["peak_explicit_live_bytes"],
    )
    assert group_sizes == [1536] * 93 + [4608] * 4
    assert result["formal_seed_addresses_accessed"] is False
    assert resource["start"] <= result["seed_address"] < (
        resource["start"] + resource["count"]
    )
    assert result["scientific_verdict"] is None
    with pytest.raises(RuntimeError, match="implementation drift"):
        validate_statistics_profile(config, result)


def test_default_stats_dispatch_uses_production_rademacher_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cnn_fpga.benchmark import phase9_powered_twin_statistics as statistics

    calls = 0

    def fast_production(*, seed, replicates, cluster_root_ids):
        nonlocal calls
        calls += 1
        return np.ones((replicates, len(cluster_root_ids)), dtype=np.int8)

    monkeypatch.setattr(statistics, "rademacher_matrix", fast_production)
    config = _config()
    result = streaming_statistics_dry_run(
        config,
        sample_callback=lambda: None,
    )
    assert calls == 97
    assert result["production_rademacher_generator_exercised"] is True
    validate_statistics_profile(config, result)

    for field in (
        "largest_density_kernel_exercised",
        "largest_density_perturbation_shape",
        "largest_density_update_bytes",
        "largest_density_trace_norm_evaluations",
        "largest_density_kernel_sha256",
        "largest_density_source_buffer_count",
        "largest_density_rss_callback_count",
        "persistent_working_set_components",
        "largest_density_peak_live_components",
        "peak_explicit_working_set_bytes",
        "peak_analysis_scratch_bytes",
        "retained_density_physicality_profile",
    ):
        mutated = dict(result)
        if field == "largest_density_kernel_exercised":
            mutated[field] = False
        elif field == "largest_density_perturbation_shape":
            mutated[field] = [199, 131, 131]
        elif field == "largest_density_kernel_sha256":
            mutated[field] = ""
        else:
            mutated[field] = 0
        with pytest.raises(RuntimeError, match="implementation drift"):
            validate_statistics_profile(config, mutated)

    mutated = copy.deepcopy(result)
    nested = mutated["retained_density_physicality_profile"]
    nested["projected_full_serial_wall_seconds"] *= 0.5
    unsigned = dict(nested)
    unsigned.pop("analysis_sha256")
    nested["analysis_sha256"] = _sha(unsigned)
    with pytest.raises(RuntimeError, match="implementation drift"):
        validate_statistics_profile(config, mutated)


def test_statistics_resource_seed_overlap_fails_before_allocation() -> None:
    config = json.loads(json.dumps(_config()))
    resource = config["seed_registry"]["resource_preflight"]
    config["seed_registry"]["joint_maxt_rademacher"]["start"] = int(
        resource["start"]
    )

    def forbidden_signs(**kwargs):
        raise AssertionError("overlapping resource seed reached allocation")

    with pytest.raises(RuntimeError, match="seed namespace overlaps"):
        streaming_statistics_dry_run(
            config,
            sign_matrix_factory=forbidden_signs,
        )


def test_resource_gate_rejects_incomplete_inventory(tmp_path: Path) -> None:
    config = _config()
    inventory = {
        "raw_status": "INCOMPLETE_FAIL_CLOSED",
        "receipt_count": 8,
        "totals": {
            "expected_rows": 5,
            "observed_rows": 4,
            "exception_rows": 1,
            "missing_rows": 1,
            "conservation_failures": 0,
        },
        "monolithic_archive": None,
        "merged_full_csv": None,
    }
    projection = {
        "projected_formal_artifact_bytes": 1,
        "projected_formal_wall_seconds_at_frozen_concurrency": 1.0,
    }
    sampling = {"peak_aggregate_rss_bytes": 1}
    decision = resource_gate_decision(
        config,
        sampling=sampling,
        projection=projection,
        inventory=inventory,
        run_directory=tmp_path,
    )
    assert decision["checks"]["inventory"] is False
    assert decision["passed"] is False


def test_resource_gate_rejects_physicality_inclusive_wall(
    tmp_path: Path,
) -> None:
    config = _config()
    inventory = {
        "raw_status": "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT",
        "receipt_count": 8,
        "totals": {
            "expected_rows": 5,
            "observed_rows": 5,
            "exception_rows": 0,
            "missing_rows": 0,
            "conservation_failures": 0,
        },
        "monolithic_archive": None,
        "merged_full_csv": None,
    }
    projection = {
        "projected_formal_artifact_bytes": 1,
        "projected_formal_wall_seconds_at_frozen_concurrency": (
            float(config["resource_contract"]["maximum_wall_seconds"])
            + 1.0
        ),
        "retained_density_physicality_serial_wall_seconds": 1.0,
    }
    decision = resource_gate_decision(
        config,
        sampling={"peak_aggregate_rss_bytes": 1},
        projection=projection,
        inventory=inventory,
        run_directory=tmp_path,
    )
    assert decision["checks"]["wall"] is False
    assert decision["passed"] is False


def test_resource_disk_gate_includes_inflight_and_analysis_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cnn_fpga.benchmark import phase9_powered_twin_preflight as preflight

    config = _config()
    reserve = int(
        config["resource_contract"]["minimum_post_projection_free_bytes"]
    )
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda path: type(
            "Usage",
            (),
            {
                "total": reserve + 10_000,
                "used": 0,
                "free": reserve + 599,
            },
        )(),
    )
    inventory = {
        "raw_status": "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT",
        "receipt_count": 8,
        "totals": {
            "expected_rows": 227_328,
            "observed_rows": 227_328,
            "exception_rows": 0,
            "missing_rows": 0,
            "conservation_failures": 0,
        },
        "monolithic_archive": None,
        "merged_full_csv": None,
    }
    decision = resource_gate_decision(
        config,
        sampling={"peak_aggregate_rss_bytes": 1},
        projection={
            "projected_formal_artifact_bytes": 100,
            "projected_formal_wall_seconds_at_frozen_concurrency": 1.0,
        },
        inventory=inventory,
        run_directory=tmp_path,
        maximum_inflight_temp_bytes=200,
        analysis_scratch_bytes=300,
    )
    assert decision["projected_post_formal_free_bytes"] == reserve - 1
    assert decision["checks"]["disk"] is False
    assert decision["passed"] is False


def test_projection_is_518_cell_component_stratified_not_uniform() -> None:
    config = _config()
    cells = build_cell_plan(config)
    measurements = []
    for plan_index in (388, 389, 403, 478, 480, 482, 484, 507):
        cell = cells[plan_index]
        measurements.append(
            {
                "plan_index": plan_index,
                "wall_seconds": 10.0 + plan_index / 1000.0,
                "object_bytes_unique": 10_000,
                "object_bytes_by_role": {
                    "round_ledger_csv": 1000,
                    "raw_iq_npy": 2000,
                    "heldout_iq_npy": 2000,
                    "primary_density_npy": (
                        3000 if cell.density_retention != "none" else 0
                    ),
                    "rb_expected_density_npy": (
                        1000 if "reset" in cell.reset_estimand_scope else 0
                    ),
                    "rb_success_probability_npy": (
                        1000 if "reset" in cell.reset_estimand_scope else 0
                    ),
                },
            }
        )
        measurements[-1]["explicit_alias_bytes"] = 0
        measurements[-1]["conservative_payload_bytes"] = sum(
            measurements[-1]["object_bytes_by_role"].values()
        )
    result = stratified_projection(
        config,
        cells,
        measurements,
        stats_wall_seconds=2.0,
        retained_density_physicality_wall_seconds=123.0,
        inventory_finalize_wall_seconds=1.0,
        inventory_profile_object_bytes=1_000,
        inventory_profile_receipt_count=8,
    )
    assert result["uniform_row_ratio_used"] is False
    assert len(result["cell_projections"]) == 518
    assert {row["plan_index"] for row in result["cell_projections"]} == set(
        range(518)
    )
    assert set(result["layers"]) == {"shared", "logical", "probe", "fault"}
    assert result["projected_formal_artifact_bytes"] > 0
    assert result["projected_formal_artifact_bytes"] == sum(
        row["projected_object_bytes"]
        for row in result["cell_projections"]
    )
    assert all(
        row["projected_transient_bytes"] == row["projected_object_bytes"]
        for row in result["cell_projections"]
    )
    mapping_rows = {
        row["plan_index"]: row["projected_mapping_anchor_bytes"]
        for row in result["cell_projections"]
    }
    assert {index for index, size in mapping_rows.items() if size > 0} == {
        0, 162, 324
    }
    raw_walls = [
        row["projected_wall_seconds"]
        for row in result["cell_projections"]
    ]
    assert result["projected_raw_lpt_wall_seconds"] >= max(raw_walls)
    assert result["projected_raw_lpt_wall_seconds"] >= (
        sum(raw_walls) / 4.0
    )
    assert result["projected_inventory_finalize_wall_seconds"] > 0.0
    assert result[
        "retained_density_physicality_serial_wall_seconds"
    ] == 123.0
    assert result["projected_formal_wall_seconds_at_frozen_concurrency"] > 125.0
    with pytest.raises(RuntimeError, match="physicality wall"):
        stratified_projection(
            config,
            cells,
            measurements,
            stats_wall_seconds=2.0,
            retained_density_physicality_wall_seconds=0.0,
            inventory_finalize_wall_seconds=1.0,
            inventory_profile_object_bytes=1_000,
            inventory_profile_receipt_count=8,
        )
