from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_powered_twin_preformal_audit as audit


def _write_self_hashed(path: Path, value: dict[str, object]) -> None:
    payload = dict(value)
    payload["analysis_sha256"] = audit._sha(payload)
    audit._immutable_json(path, payload)


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
    plan_sha = "2" * 64
    source_sha = "3" * 64
    contract = {
        "schema_version": "PHASE9-POWERED-TWIN-CONTRACT-PREFLIGHT-V2",
        "status": "PASS_OUTCOME_FREE_CONTRACT_PREFLIGHT",
        "plan_summary": {"plan_sha256": plan_sha},
        "scientific_execution_released": False,
    }
    _write_self_hashed(tmp_path / paths["contract_preflight"], contract)
    resource: dict[str, object] = {
        "verdict": "PASS_RESOURCE_PREFLIGHT",
        "source_snapshot_sha256": source_sha,
        "config_sha256": config_sha,
        "plan_sha256": plan_sha,
        "full_size_receipt_count": 5,
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
    plan = {
        "canonical_plan_sha256": plan_sha,
        "cell_count": 518,
        "row_count": 2_085_888,
        "primary_density_count": 482_304,
    }

    monkeypatch.setattr(
        audit,
        "load_config",
        lambda root: (copy.deepcopy(config), {"sha256": config_sha}),
    )
    monkeypatch.setattr(audit, "plan_payload", lambda current: dict(plan))
    monkeypatch.setattr(
        audit,
        "runtime_source_snapshot",
        lambda root, current: dict(snapshot),
    )

    def fake_validation(
        root: Path,
        current: dict[str, object],
        current_snapshot: dict[str, object],
    ) -> dict[str, object]:
        report = {
            "schema_version": audit.VALIDATION_SCHEMA,
            "source_snapshot_sha256": source_sha,
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
