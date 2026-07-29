from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
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
    _sha,
    _immutable_json,
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


def test_frozen_full_profile_is_four_b_workers_plus_one_a() -> None:
    config = _config()
    cells = build_cell_plan(config)
    concurrent, singleton = profile_cells(config, cells)
    assert [cell.plan_index for cell in concurrent] == [389, 403, 507, 485]
    assert [cell.sample_count for cell in concurrent] == [1536, 1536, 1536, 4608]
    assert [cell.layer for cell in concurrent] == [
        "shared",
        "logical",
        "probe",
        "fault",
    ]
    assert singleton.plan_index == 388
    assert singleton.backend == "A"
    assert singleton.sample_count == 1536
    mutated = copy.deepcopy(config)
    mutated["resource_contract"]["profile_plan"]["four_worker_concurrent_peak"][
        "full_frozen_denominator"
    ] = False
    with pytest.raises(RuntimeError, match="full frozen denominator"):
        profile_cells(mutated, cells)


def test_endpoint_only_and_insufficient_concurrency_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="endpoint-only"):
        validate_continuous_sampling(_sampling(count=2, active=2))
    with pytest.raises(RuntimeError, match="endpoint-only"):
        validate_continuous_sampling(_sampling(count=3, active=1))
    with pytest.raises(RuntimeError, match="concurrency"):
        validate_continuous_sampling(_sampling(overlap=3))
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
        sample_interval_seconds=0.01,
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
            runtime_fingerprint={"fixture": True},
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
        "receipt_count": 5,
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
        "receipt_count": 5,
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


def test_projection_is_518_cell_component_stratified_not_uniform() -> None:
    config = _config()
    cells = build_cell_plan(config)
    measurements = []
    for plan_index in (388, 389, 403, 485, 507):
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
    result = stratified_projection(
        config,
        cells,
        measurements,
        stats_wall_seconds=2.0,
        retained_density_physicality_wall_seconds=123.0,
    )
    assert result["uniform_row_ratio_used"] is False
    assert len(result["cell_projections"]) == 518
    assert {row["plan_index"] for row in result["cell_projections"]} == set(
        range(518)
    )
    assert set(result["layers"]) == {"shared", "logical", "probe", "fault"}
    assert result["projected_formal_artifact_bytes"] > 0
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
        )
