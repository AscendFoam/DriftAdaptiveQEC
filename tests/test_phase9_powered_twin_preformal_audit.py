from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_powered_twin_preformal_audit as audit
from cnn_fpga.benchmark.phase9_powered_twin_preflight import _record_attempt


ROOT = Path(__file__).resolve().parents[1]


def _write_self_hashed(path: Path, value: dict[str, object]) -> None:
    payload = dict(value)
    payload["analysis_sha256"] = audit._sha(payload)
    audit._immutable_json(path, payload)


def _pass_resource_report() -> dict[str, object]:
    report: dict[str, object] = {
        field: None for field in audit.RESOURCE_REPORT_FIELDS
    }
    report.update(
        {
            "schema_version": audit.PREFLIGHT_SCHEMA,
            "task_id": audit.TASK_ID,
            "run_id": "resource_fixture",
            "runner_id": audit.RESOURCE_RUNNER_ID,
            "verdict": audit.PASS_VERDICT,
            "config_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "source_snapshot_sha256": "3" * 64,
            "formal_artifact_namespace_accessed": False,
            "scientific_verdict": None,
            "qualified_claim": None,
            "claim_boundary": {
                field: None for field in audit.EXPECTED_CLAIM_FIELDS
            },
        }
    )
    report.pop("analysis_sha256")
    report["analysis_sha256"] = audit._sha(report)
    return report


def _sampling_fixture(
    root: Path,
    preflight_root: Path,
    *,
    wrong_aggregate: bool = False,
    stage_regression: bool = False,
) -> dict[str, object]:
    stages = [
        "starting",
        "formal_lpt_four_worker_peak",
        "representative_four_worker_profiles",
        "joint_maxt_3037x199",
        "inventory_finalize_no_copy",
    ]
    if stage_regression:
        stages[2], stages[3] = stages[3], stages[2]
    times = [0.0, 5.0, 10.0, 15.0, 35.0]
    previous = "0" * 64
    records: list[dict[str, object]] = []
    for sequence, (stage, monotonic) in enumerate(zip(stages, times)):
        child_pids = (
            [101, 102, 103, 104]
            if stage == "formal_lpt_four_worker_peak"
            else [201, 202, 203, 204]
            if stage == "representative_four_worker_profiles"
            else []
        )
        child_rss = {str(pid): 10 + pid for pid in child_pids}
        child_trees = {str(pid): [pid] for pid in child_pids}
        aggregate = 100 + sum(child_rss.values())
        if wrong_aggregate and sequence == 1:
            aggregate -= 1
        record: dict[str, object] = {
            "schema_version": audit.SAMPLING_SCHEMA,
            "sequence": sequence,
            "monotonic_seconds": monotonic,
            "parent_pid": 99,
            "parent_rss_bytes": 100,
            "child_rss_bytes": child_rss,
            "child_process_tree_pids": child_trees,
            "live_child_count": len(child_rss),
            "aggregate_rss_bytes": aggregate,
            "stage": stage,
            "previous_sample_sha256": previous,
        }
        record["sample_sha256"] = audit._sha(record)
        previous = str(record["sample_sha256"])
        records.append(record)
    path = preflight_root / "resource_samples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(audit._canonical(record) + b"\n" for record in records)
    )
    stage_peaks: dict[str, int] = {}
    peak: dict[str, object] | None = None
    peak_rss = 0
    for record in records:
        stage = str(record["stage"])
        rss = int(record["aggregate_rss_bytes"])
        stage_peaks[stage] = max(stage_peaks.get(stage, 0), rss)
        if rss >= peak_rss:
            peak_rss = rss
            peak = record
    summary: dict[str, object] = {
        "schema_version": audit.SAMPLING_SCHEMA,
        "sample_count": len(records),
        "active_child_sample_count": sum(
            bool(record["child_rss_bytes"]) for record in records
        ),
        "peak_aggregate_rss_bytes": peak_rss,
        "maximum_observed_live_children": 4,
        "stage_peak_aggregate_rss_bytes": stage_peaks,
        "first_sample": records[0],
        "last_sample": records[-1],
        "peak_sample": peak,
        "sample_chain_tip_sha256": previous,
        "evidence": audit._binding(path, root),
    }
    summary["summary_sha256"] = audit._sha(summary)
    return {
        "sampling": summary,
        "sample_interval_seconds": 5.0,
        "resource_sample_count": len(records),
        "actual_peak_concurrency": 4,
        "maximum_observed_worker_overlap": 4,
    }


def _isolated_release_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resource_mutation: tuple[str, object] | None = None,
) -> dict[str, object]:
    paths = {
        "run_directory": "runs/formal",
        "object_store": "runs/formal/objects/sha256",
        "staging_directory": "runs/formal/staging",
        "receipt_directory": "runs/formal/receipts",
        "plan": "runs/plan.json",
        "seed_registry": "runs/seed_registry.json",
        "historical_seed_scan": "runs/historical_seed_scan.json",
        "contract_preflight": "docs/contract.json",
        "resource_preflight": "runs/resource.json",
        "preformal_validation": "runs/validation.json",
        "preformal_seal": "docs/seal.json",
        "inventory": "docs/inventory.json",
        "execution_manifest": "docs/manifest.json",
        "independent_verification": "docs/verification.json",
    }
    config = {
        "task_id": "T-RISK-20260728-04",
        "artifact_paths": paths,
        "runtime_contract": {
            "preimport_thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        },
        "runtime_sources": {"validation_paths": []},
    }
    config_sha = "1" * 64
    config_binding = {
        "path": "configs/config.json",
        "bytes": 123,
        "sha256": config_sha,
    }
    plan_sha = "2" * 64
    source_sha = "3" * 64
    plan = {
        "canonical_plan_sha256": plan_sha,
        "cell_count": 518,
        "row_count": 2_085_888,
        "primary_density_count": 482_304,
    }
    seed_registry = {"registry_sha256": "4" * 64}
    historical_scan = {"scan_manifest_sha256": "5" * 64}
    contract = {
        "task_id": audit.TASK_ID,
        "schema_version": "PHASE9-POWERED-TWIN-CONTRACT-PREFLIGHT-V5",
        "status": "PASS_OUTCOME_FREE_CONTRACT_PREFLIGHT",
        "plan_summary": {"plan_sha256": plan_sha},
        "formal_outcomes_accessed": False,
        "scientific_execution_released": False,
        "qualified_claim": None,
        "claim_boundary": {
            field: None for field in audit.EXPECTED_CLAIM_FIELDS
        },
        "source_registry_summary": {
            "source_snapshot_sha256": source_sha,
            "runtime_source_count": 22,
            "validation_source_count": 9,
            "all_registered_sources_live_and_regular": True,
        },
        "gates": {
            "C01_all_parent_bytes_verified": True,
            "C02_t03_t05_t06_semantic_chain_verified": True,
            "C03_exact_518_cell_plan": True,
            "C04_exact_2085888_row_denominator": True,
            "C05_exact_482304_primary_densities": True,
            "C06_fault_state_major_6x768": True,
            "C07_historical_seed_scan_recomputed": True,
            "C08_actual_seed_addresses_injective": True,
            "C09_content_addressed_no_zip_contract_frozen": True,
            "C10_all_claims_null_and_scientific_execution_blocked": True,
            "C11_all_runtime_and_validation_sources_live_regular": True,
        },
        "parent_semantic_checks": {
            "P01_t03_design_repair_independent_pass": True,
            "P02_t05_statistical_no_go_preserved": True,
            "P03_t06_independent_count_pass": True,
            "P04_selected_count_exact": True,
            "P05_selected_blueprint_exact": True,
            "P06_blueprint_counts_consume_selected_count": True,
            "P07_parent_claims_remain_null": True,
        },
        "bindings": {},
    }
    for name, relative, value in (
        ("plan", paths["plan"], plan),
        ("seed_registry", paths["seed_registry"], seed_registry),
        (
            "historical_seed_scan",
            paths["historical_seed_scan"],
            historical_scan,
        ),
    ):
        target = tmp_path / relative
        audit._immutable_json(target, value)
        contract["bindings"][name] = audit._binding(target, tmp_path)
    contract["bindings"]["config"] = config_binding
    _write_self_hashed(tmp_path / paths["contract_preflight"], contract)
    resource: dict[str, object] = {
        "run_id": "resource_fixture",
        "verdict": "PASS_RESOURCE_PREFLIGHT",
        "source_snapshot_sha256": source_sha,
        "config_sha256": config_sha,
        "plan_sha256": plan_sha,
        "full_size_receipt_count": 8,
        "scientific_verdict": None,
        "qualified_claim": None,
        "claim_boundary": {
            field: None for field in audit.EXPECTED_CLAIM_FIELDS
        },
        "maximum_observed_worker_overlap": 4,
        "joint_maxt_profile": {"gate_count": 3037, "replicates": 199},
        "inventory": {
            "monolithic_archive": None,
            "merged_full_csv": None,
        },
        "resource_sample_count": 3,
    }
    if resource_mutation is not None:
        resource[resource_mutation[0]] = resource_mutation[1]
    _write_self_hashed(tmp_path / paths["resource_preflight"], resource)

    snapshot = {
        "source_snapshot_sha256": source_sha,
        "runtime_source_count": 22,
        "validation_source_count": 9,
    }
    monkeypatch.setattr(
        audit,
        "load_config",
        lambda root: (
            copy.deepcopy(config),
            copy.deepcopy(config_binding),
        ),
    )
    monkeypatch.setattr(audit, "plan_payload", lambda current: dict(plan))
    monkeypatch.setattr(
        audit,
        "seed_registry_payload",
        lambda current: dict(seed_registry),
    )
    monkeypatch.setattr(
        audit,
        "historical_seed_scan",
        lambda root, config_path: dict(historical_scan),
    )
    monkeypatch.setattr(
        audit,
        "runtime_source_snapshot",
        lambda root, current: dict(snapshot),
    )
    attempt_root = tmp_path / "runs/resource_fixture"
    attempt_root.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("attempts.jsonl", b'{"fixture":"attempts"}\n'),
        ("attempt_events/00000000.json", b'{"fixture":"start"}\n'),
        ("attempt_events/00000001.json", b'{"fixture":"pass"}\n'),
    ):
        target = attempt_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)

    def fake_resource_validation(
        root: Path,
        current: dict[str, object],
        current_binding: dict[str, object],
        current_plan: dict[str, object],
        current_snapshot: dict[str, object],
        resource_path: Path,
        resource_report: dict[str, object],
    ) -> dict[str, object]:
        claims = resource_report.get("claim_boundary")
        if (
            resource_report.get("scientific_verdict") is not None
            or resource_report.get("qualified_claim") is not None
            or not isinstance(claims, dict)
            or any(value is not None for value in claims.values())
        ):
            raise RuntimeError("resource preflight claim boundary drift")
        attempts = {
            "binding": audit._binding(
                attempt_root / "attempts.jsonl", root
            ),
            "start_witness": audit._binding(
                attempt_root / "attempt_events/00000000.json", root
            ),
            "pass_witness": audit._binding(
                attempt_root / "attempt_events/00000001.json", root
            ),
        }
        evidence: dict[str, object] = {
            "schema_version": audit.RESOURCE_CONSUMPTION_SCHEMA,
            "ledger_rows_verified": 227_328,
            "reset_rows_verified": 15_360,
            "formal_seed_addresses_accessed": False,
            "attempt_chain": attempts,
            "claim_boundary": {
                field: None for field in audit.EXPECTED_CLAIM_FIELDS
            },
        }
        evidence["analysis_sha256"] = audit._sha(evidence)
        return evidence

    monkeypatch.setattr(
        audit,
        "validate_resource_release_evidence",
        fake_resource_validation,
    )

    def fake_validation(
        root: Path,
        current: dict[str, object],
        current_snapshot: dict[str, object],
        *,
        resource_binding: dict[str, object],
        resource_consumption: dict[str, object],
    ) -> dict[str, object]:
        report = {
            "schema_version": audit.VALIDATION_SCHEMA,
            "source_snapshot_sha256": source_sha,
            "resource_preflight": resource_binding,
            "resource_consumption_sha256": resource_consumption[
                "analysis_sha256"
            ],
            "attempt_ledger": resource_consumption["attempt_chain"][
                "binding"
            ],
            "returncode": 0,
            "verdict": "PASS_FOCUSED_ANTISIMPLIFICATION",
            "claim_boundary": {
                field: None for field in audit.EXPECTED_CLAIM_FIELDS
            },
        }
        report["analysis_sha256"] = audit._sha(report)
        audit._immutable_json(
            root / paths["preformal_validation"],
            report,
        )
        return report

    monkeypatch.setattr(audit, "run_focused_validation", fake_validation)
    return config


def test_immutable_preformal_publication_rejects_conflict(tmp_path: Path) -> None:
    path = tmp_path / "seal.json"
    audit._immutable_json(path, {"value": 1})
    audit._immutable_json(path, {"value": 1})
    with pytest.raises(RuntimeError, match="conflicting immutable"):
        audit._immutable_json(path, {"value": 2})


def test_preformal_release_requires_empty_formal_transaction(
    tmp_path: Path,
) -> None:
    config = {
        "artifact_paths": {
            "object_store": "objects",
            "staging_directory": "staging",
            "receipt_directory": "receipts",
            "inventory": "inventory.json",
            "execution_manifest": "manifest.json",
            "independent_verification": "verification.json",
        }
    }
    audit._assert_no_formal_outcome(tmp_path, config)
    object_path = tmp_path / "objects/sha256/aa/object"
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b"formal")
    with pytest.raises(RuntimeError, match="formal content object"):
        audit._assert_no_formal_outcome(tmp_path, config)


def test_preformal_seal_releases_only_raw_execution_and_keeps_claims_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolated_release_fixture(tmp_path, monkeypatch)
    seal = audit.create_preformal_seal(tmp_path)
    assert seal["verdict"] == "PASS_PREFORMAL_RELEASE"
    assert seal["raw_execution_released"] is True
    assert seal["scientific_verdict_released"] is False
    assert seal["scientific_verdict"] is None
    assert seal["qualified_claim"] is None
    assert seal["official_puviani_surpass"] is None
    assert set(seal["claim_boundary"].values()) == {None}
    assert all(seal["gates"].values())
    published = audit._strict_json(tmp_path / "docs/seal.json")
    audit._verify_self_hash(published)


def test_preformal_seal_rejects_nonnull_resource_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolated_release_fixture(
        tmp_path,
        monkeypatch,
        resource_mutation=("qualified_claim", "surpass"),
    )
    with pytest.raises(RuntimeError, match="resource preflight"):
        audit.create_preformal_seal(tmp_path)


@pytest.mark.parametrize("version", ("V1", "V2", "V3", "V4"))
def test_preformal_seal_rejects_superseded_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    config = _isolated_release_fixture(tmp_path, monkeypatch)
    path = tmp_path / config["artifact_paths"]["contract_preflight"]
    contract = audit._strict_json(path)
    contract.pop("analysis_sha256")
    contract["schema_version"] = (
        f"PHASE9-POWERED-TWIN-CONTRACT-PREFLIGHT-{version}"
    )
    path.unlink()
    _write_self_hashed(path, contract)
    with pytest.raises(RuntimeError, match="V5 outcome-free"):
        audit.create_preformal_seal(tmp_path)


def test_preformal_seal_rejects_stale_contract_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _isolated_release_fixture(tmp_path, monkeypatch)
    path = tmp_path / config["artifact_paths"]["contract_preflight"]
    contract = audit._strict_json(path)
    contract.pop("analysis_sha256")
    contract["source_registry_summary"]["source_snapshot_sha256"] = "4" * 64
    path.unlink()
    _write_self_hashed(path, contract)
    with pytest.raises(RuntimeError, match="live bindings"):
        audit.create_preformal_seal(tmp_path)


def test_preformal_seal_rejects_stale_contract_plan_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _isolated_release_fixture(tmp_path, monkeypatch)
    plan_path = tmp_path / config["artifact_paths"]["plan"]
    plan_path.unlink()
    audit._immutable_json(plan_path, {"mutated": True})
    with pytest.raises(RuntimeError, match="live bindings"):
        audit.create_preformal_seal(tmp_path)


@pytest.mark.parametrize(
    ("artifact_name", "replacement"),
    (
        ("plan", {"mutated": "plan"}),
        ("seed_registry", {"mutated": "registry"}),
        ("historical_seed_scan", {"mutated": "scan"}),
    ),
)
def test_preformal_seal_rejects_semantic_artifact_and_rebound_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    replacement: dict[str, str],
) -> None:
    config = _isolated_release_fixture(tmp_path, monkeypatch)
    paths = config["artifact_paths"]
    artifact_path = tmp_path / paths[artifact_name]
    artifact_path.unlink()
    audit._immutable_json(artifact_path, replacement)
    contract_path = tmp_path / paths["contract_preflight"]
    contract = audit._strict_json(contract_path)
    contract.pop("analysis_sha256")
    contract["bindings"][artifact_name] = audit._binding(
        artifact_path, tmp_path
    )
    contract_path.unlink()
    _write_self_hashed(contract_path, contract)
    with pytest.raises(RuntimeError, match="live bindings"):
        audit.create_preformal_seal(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("task_id", "WRONG-TASK"),
        ("parent_semantic_checks", {}),
    ),
)
def test_preformal_seal_rejects_contract_identity_or_parent_gate_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    config = _isolated_release_fixture(tmp_path, monkeypatch)
    path = tmp_path / config["artifact_paths"]["contract_preflight"]
    contract = audit._strict_json(path)
    contract.pop("analysis_sha256")
    contract[field] = value
    path.unlink()
    _write_self_hashed(path, contract)
    with pytest.raises(RuntimeError, match="live bindings"):
        audit.create_preformal_seal(tmp_path)


def test_resource_report_requires_exact_pass_schema_and_null_claims() -> None:
    report = _pass_resource_report()
    assert (
        audit._require_pass_resource_report(
            report,
            config_sha256="1" * 64,
            plan_sha256="2" * 64,
            source_snapshot_sha256="3" * 64,
        )
        == "resource_fixture"
    )
    extra = dict(report)
    extra["hidden"] = True
    extra.pop("analysis_sha256")
    extra["analysis_sha256"] = audit._sha(extra)
    with pytest.raises(RuntimeError, match="top-level schema"):
        audit._require_pass_resource_report(
            extra,
            config_sha256="1" * 64,
            plan_sha256="2" * 64,
            source_snapshot_sha256="3" * 64,
        )
    claimed = dict(report)
    claimed["qualified_claim"] = "surpass"
    claimed.pop("analysis_sha256")
    claimed["analysis_sha256"] = audit._sha(claimed)
    with pytest.raises(RuntimeError, match="claim boundary"):
        audit._require_pass_resource_report(
            claimed,
            config_sha256="1" * 64,
            plan_sha256="2" * 64,
            source_snapshot_sha256="3" * 64,
        )


@pytest.mark.parametrize(
    "relative",
    (
        "runs/t04_resource_preflight_full_20260730_0655/"
        "resource_preflight_failed.json",
        "runs/t04_resource_preflight_full_v4_20260730_080203/"
        "resource_preflight_failed.json",
    ),
)
def test_real_failed_resource_transactions_are_never_releasable(
    relative: str,
) -> None:
    report = audit._strict_json(ROOT / relative)
    with pytest.raises(RuntimeError):
        audit._require_pass_resource_report(
            report,
            config_sha256=str(report["config_sha256"]),
            plan_sha256=str(report["plan_sha256"]),
            source_snapshot_sha256=str(
                report["source_snapshot_sha256"]
            ),
        )


def test_sampling_raw_chain_is_recomputed_and_coordinated_forgery_fails(
    tmp_path: Path,
) -> None:
    preflight_root = tmp_path / "runs/t04_resource_preflight_fixture"
    resource = _sampling_fixture(tmp_path, preflight_root)
    summary, concurrent, singleton, stage_windows = (
        audit._verify_sampling_evidence(
        tmp_path,
        preflight_root,
        resource,
        heartbeat_period_seconds=30.0,
        )
    )
    assert summary["sample_count"] == 5
    assert concurrent == {101, 102, 103, 104}
    assert singleton == {201, 202, 203, 204}
    assert stage_windows == {
        "starting": 5.0,
        "formal_lpt_four_worker_peak": 5.0,
        "representative_four_worker_profiles": 5.0,
        "joint_maxt_3037x199": 20.0,
    }

    forged_root = tmp_path / "runs/t04_resource_preflight_forged"
    forged = _sampling_fixture(
        tmp_path,
        forged_root,
        wrong_aggregate=True,
    )
    with pytest.raises(RuntimeError, match="RSS arithmetic"):
        audit._verify_sampling_evidence(
            tmp_path,
            forged_root,
            forged,
            heartbeat_period_seconds=30.0,
        )

    regressed_root = tmp_path / "runs/t04_resource_preflight_regressed"
    regressed = _sampling_fixture(
        tmp_path,
        regressed_root,
        stage_regression=True,
    )
    with pytest.raises(RuntimeError, match="stage regressed"):
        audit._verify_sampling_evidence(
            tmp_path,
            regressed_root,
            regressed,
            heartbeat_period_seconds=30.0,
        )


def test_attempt_chain_requires_exact_start_pass_and_report_binding(
    tmp_path: Path,
) -> None:
    preflight_root = tmp_path / "runs/t04_resource_preflight_fixture"
    attempt_path = preflight_root / "attempts.jsonl"
    namespace = {
        "object_store": (
            "runs/t04_resource_preflight_fixture/objects/sha256"
        ),
        "staging_directory": (
            "runs/t04_resource_preflight_fixture/staging"
        ),
        "receipt_directory": (
            "runs/t04_resource_preflight_fixture/receipts"
        ),
    }
    resource = {
        "run_id": "fixture",
        "artifact_namespace": namespace,
        "analysis_sha256": "a" * 64,
    }
    heartbeat = {
        "owner_token": "b" * 32,
        "pid": 99,
        "process_creation_time": 1234.5,
    }
    _record_attempt(
        attempt_path,
        task_id=audit.TASK_ID,
        run_id="fixture",
        event="START_RESOURCE_PREFLIGHT",
        payload={
            "formal_seed_addresses_accessed": False,
            "artifact_namespace": namespace,
            "owner_token": heartbeat["owner_token"],
            "owner_pid": heartbeat["pid"],
            "process_creation_time": heartbeat[
                "process_creation_time"
            ],
        },
    )
    resource["attempt_witnesses_before_terminal"] = [
        audit._binding(
            preflight_root / "attempt_events/00000000.json",
            tmp_path,
        )
    ]
    _record_attempt(
        attempt_path,
        task_id=audit.TASK_ID,
        run_id="fixture",
        event="PASS_RESOURCE_PREFLIGHT",
        payload={
            "analysis_sha256": "a" * 64,
            "formal_seed_addresses_accessed": False,
        },
    )
    evidence = audit._verify_attempt_chain(
        tmp_path, preflight_root, resource, heartbeat
    )
    assert evidence["event_count"] == 2
    lines = attempt_path.read_bytes().splitlines()
    second = audit._strict_json_bytes(lines[1], label="fixture pass")
    second["payload"]["analysis_sha256"] = "b" * 64
    second.pop("event_sha256")
    second["event_sha256"] = audit._sha(second)
    forged_line = audit._canonical(second) + b"\n"
    attempt_path.write_bytes(lines[0] + b"\n" + forged_line)
    (
        preflight_root / "attempt_events/00000001.json"
    ).write_bytes(forged_line)
    with pytest.raises(RuntimeError, match="START/PASS semantic"):
        audit._verify_attempt_chain(
            tmp_path, preflight_root, resource, heartbeat
        )


def test_preformal_validation_cannot_be_reused_across_resource_reports(
    tmp_path: Path,
) -> None:
    config = {
        "artifact_paths": {
            "preformal_validation": "validation.json",
            "run_directory": "runs/formal",
        },
        "runtime_sources": {"validation_paths": []},
        "runtime_contract": {
            "preimport_thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        },
    }
    snapshot = {"source_snapshot_sha256": "1" * 64}
    resource_a = {
        "path": "resource-a.json",
        "bytes": 1,
        "sha256": "2" * 64,
    }
    attempt = {
        "path": "attempts-a.jsonl",
        "bytes": 1,
        "sha256": "3" * 64,
    }
    consumption = {
        "analysis_sha256": "4" * 64,
        "attempt_chain": {"binding": attempt},
    }
    report: dict[str, object] = {
        "schema_version": audit.VALIDATION_SCHEMA,
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "resource_preflight": resource_a,
        "resource_consumption_sha256": consumption["analysis_sha256"],
        "attempt_ledger": attempt,
        "command": ["fixture"],
        "python": list(audit.sys.version_info[:3]),
        "platform": audit.platform.platform(),
        "returncode": 0,
        "elapsed_seconds": 1.0,
        "stdout": "pass",
        "stderr": "",
        "stdout_sha256": audit.sha256(b"pass").hexdigest(),
        "stderr_sha256": audit.sha256(b"").hexdigest(),
        "verdict": "PASS_FOCUSED_ANTISIMPLIFICATION",
        "formal_outcomes_accessed": False,
        "claim_boundary": {
            field: None for field in audit.EXPECTED_CLAIM_FIELDS
        },
    }
    report["analysis_sha256"] = audit._sha(report)
    audit._immutable_json(tmp_path / "validation.json", report)
    observed = audit.run_focused_validation(
        tmp_path,
        config,
        snapshot,
        resource_binding=resource_a,
        resource_consumption=consumption,
    )
    assert observed["resource_preflight"] == resource_a

    resource_b = {
        "path": "resource-b.json",
        "bytes": 1,
        "sha256": "5" * 64,
    }
    with pytest.raises(RuntimeError, match="not reusable"):
        audit.run_focused_validation(
            tmp_path,
            config,
            snapshot,
            resource_binding=resource_b,
            resource_consumption=consumption,
        )


def test_measurement_wall_cannot_be_coordinately_shrunk_below_stage_span(
) -> None:
    indices = [388, 389, 403, 478, 480, 482, 484, 507]
    peak_indices = {478, 480, 482, 484}
    receipts: dict[int, dict[str, object]] = {}
    measurements: list[dict[str, object]] = []
    peak_pid = iter((101, 102, 103, 104))
    representative_pid = iter((201, 202, 203, 204))
    for index in indices:
        receipt = {
            "cell": {
                "chunk_id": f"chunk-{index}",
                "plan_index": index,
                "layer": "fault" if index in peak_indices else "shared",
            },
            "objects": [
                {
                    "role": "round_ledger_csv",
                    "bytes": 100,
                    "sha256": f"{index:064x}"[-64:],
                }
            ],
            "diagnostics": {"expected_rows": 1, "reset_rows": 0},
            "receipt_sha256": f"{index + 1:064x}"[-64:],
        }
        receipts[index] = receipt
        measurement = audit._measurement_from_receipt(receipt)
        is_peak = index in peak_indices
        measurement.update(
            {
                "pid": next(peak_pid if is_peak else representative_pid),
                "wall_seconds": 90.0,
                "profile_peak_aggregate_rss_bytes": (
                    1000 if is_peak else 2000
                ),
            }
        )
        measurements.append(measurement)
    resource = {"profile_measurements": measurements}
    sampling = {
        "stage_peak_aggregate_rss_bytes": {
            "formal_lpt_four_worker_peak": 1000,
            "representative_four_worker_profiles": 2000,
        }
    }
    audit._verify_measurements(
        resource,
        receipts,
        sampling,
        concurrent_pids={101, 102, 103, 104},
        representative_pids={201, 202, 203, 204},
        stage_windows={
            "formal_lpt_four_worker_peak": 100.0,
            "representative_four_worker_profiles": 100.0,
        },
    )
    forged = copy.deepcopy(resource)
    for item in forged["profile_measurements"]:
        item["wall_seconds"] = 1.0
    with pytest.raises(RuntimeError, match="stage wall binding"):
        audit._verify_measurements(
            forged,
            receipts,
            sampling,
            concurrent_pids={101, 102, 103, 104},
            representative_pids={201, 202, 203, 204},
            stage_windows={
                "formal_lpt_four_worker_peak": 100.0,
                "representative_four_worker_profiles": 100.0,
            },
        )
