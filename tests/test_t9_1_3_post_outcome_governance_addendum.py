from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import tarfile

import pytest

from cnn_fpga.benchmark import (
    t9_1_3_post_outcome_governance_addendum as subject,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/t9_1_3_post_outcome_governance_addendum.json"
SOURCE = ROOT / "docs/t9_1_3_post_outcome_governance_source_data.csv"
CONTRACT = ROOT / "configs/phase9/t9_1_4_input_contract.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published() -> dict[str, object]:
    return _load(REPORT)


def _write_live_no_go_fixture(directory: Path) -> Path:
    transaction_id = "fixture-transaction-0001"
    reason_code = "FINALIZATION_FAILED_NO_VALID_SEAL"
    payload = directory / "failure.txt"
    payload.write_text("synthetic fail-closed evidence\n", encoding="utf-8")
    archive = directory / "failed_transaction.tar"
    with tarfile.open(archive, "w") as stream:
        stream.add(payload, arcname="failure.txt")
    inventory = directory / "failure_inventory.json"
    inventory_value = {
        "schema_version": "t9.1.3-failed-finalization-archive-v1",
        "transaction_id": transaction_id,
        "terminal_state": "FAILED_FAIL_CLOSED",
        "failure_marker_status": reason_code,
        "valid_pass_seal": False,
        "source_file_count": 1,
    }
    inventory.write_text(
        json.dumps(inventory_value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    archive_binding = subject._binding(archive)
    inventory_binding = subject._binding(inventory)
    archive_seal = directory / "archive_seal.json"
    archive_seal_value = {
        "schema_version": "t9.1.3-failed-finalization-tar-seal-v1",
        "archive_path": archive_binding["path"],
        "archive_bytes": archive_binding["bytes"],
        "archive_sha256": archive_binding["sha256"],
        "inventory_path": inventory_binding["path"],
        "inventory_sha256": inventory_binding["sha256"],
        "source_file_count": 1,
    }
    archive_seal.write_text(
        json.dumps(archive_seal_value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "schema_version": subject.FAILURE_MANIFEST_SCHEMA_VERSION,
        "task_id": subject.TASK_ID,
        "terminal_state": "FAILED_FAIL_CLOSED",
        "terminal_result": subject.NO_GO_TERMINAL_STATE,
        "observed_at_utc": "2026-07-24T00:00:00+00:00",
        "transaction_id": transaction_id,
        "reason": {
            "code": reason_code,
            "stage": "FINALIZATION",
            "message": "A test finalizer failed closed and emitted no valid seal.",
        },
        "failure_evidence": {
            "archive": archive_binding,
            "archive_seal": subject._binding(archive_seal),
            "inventory": inventory_binding,
            "archive_hash_verification": (
                "All three fixture evidence files are live raw-hash bound."
            ),
        },
        "typed_null_payload": {
            "selected_controller": None,
            "numeric_metrics": None,
            "rank": None,
            "official_exact": None,
            "puviani_surpass": None,
            "paper_scale_lifetime": None,
        },
        "semantic_sha256": "",
    }
    manifest["semantic_sha256"] = (
        subject._failure_manifest_semantic_sha256(manifest)
    )
    manifest_path = directory / "failure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_published_addendum_passes_independent_live_verification() -> None:
    checks = subject.validate_report(REPORT, verify_live_files=True)
    assert checks
    assert all(checks.values())
    downstream_checks = subject.validate_downstream_handoff(REPORT)
    assert all(downstream_checks.values())
    report = _published()
    assert report["schema_version"] == subject.SCHEMA_VERSION
    assert report["terminal_branch"] == subject.QUALIFIED_BRANCH
    assert report["failure_manifest"] is None
    assert report["status"] == subject.STATUS
    assert report["verdict"] == subject.VERDICT_PASS
    assert report["gate_summary"] == {
        "passed": 16,
        "total": 16,
        "failed": [],
    }
    assert report["downstream_semantic_seal"] == (
        subject._downstream_semantic_seal(report)
    )
    assert report["downstream_semantic_seal"]["consumer_rule"] == (
        subject.DOWNSTREAM_CONSUMER_RULE
    )
    contract = _load(CONTRACT)
    assert contract["downstream_addendum_semantic_contract"][
        "consumer_rule"
    ] == subject.DOWNSTREAM_CONSUMER_RULE
    assert contract["no_go_failure_manifest_fixture_policy"] == (
        subject.NO_GO_FIXTURE_POLICY
    )


def test_parent_report_config_deviation_and_terminal_ledgers_are_raw_bound() -> None:
    report = _published()
    contract = _load(CONTRACT)
    assert report["parent_bindings"] == contract["parent_artifacts"]
    assert report["parent_bindings"] == subject.EXPECTED_PARENT_ARTIFACTS
    for binding in report["parent_bindings"].values():
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
    assert (
        report["parent_bindings"]["parent_report"]["sha256"]
        == subject.PARENT_REPORT_RAW_SHA256
    )
    assert (
        report["parent_snapshot"]["analysis_sha256"]
        == subject.PARENT_REPORT_ANALYSIS_SHA256
    )
    assert report["parent_snapshot"]["published_gate_count"] == 52
    assert report["parent_snapshot"]["detected_mutation_count"] == 81


def test_D43_recomputes_every_epoch_and_proves_clipping_was_inactive_only_here() -> None:
    gradient = _published()["gradient_clipping_reanalysis"]
    assert gradient["max_norm"] == 10.0
    assert gradient["row_census"] == {
        "total_rows": 40000,
        "agent_count": 40,
        "agents_per_family": {"mf": 20, "nmf": 20},
        "epochs_per_agent": 1000,
        "all_agent_epoch_sets_exact": True,
    }
    assert gradient["scopes"]["all"]["rows"] == 40000
    assert gradient["scopes"]["mf"]["rows"] == 20000
    assert gradient["scopes"]["nmf"]["rows"] == 20000
    assert gradient["scopes"]["all"]["nonfinite_rows"] == 0
    assert gradient["scopes"]["all"]["nonpositive_rows"] == 0
    assert (
        gradient["scopes"]["all"][
            "rows_scaled_by_pytorch_clip_rule"
        ]
        == 0
    )
    assert gradient["scopes"]["all"]["rows_strictly_gt_max_norm"] == 0
    assert (
        gradient["scopes"]["all"]["minimum_unclamped_scale_coefficient"]
        > 1.0
    )
    assert gradient["scopes"]["all"]["maximum"] == 6.72778063230222
    assert gradient["scopes"]["mf"]["maximum"] == 5.200336572564351
    assert gradient["scopes"]["nmf"]["maximum"] == 6.72778063230222
    assert "does not establish counterfactual equivalence" in gradient[
        "observed_run_conclusion"
    ]


def test_D43_D46_are_material_nonpromotional_deviations_with_exact_census() -> None:
    report = _published()
    rows = {row["id"]: row for row in report["addendum_deviations"]}
    assert list(rows) == ["D43", "D44", "D45", "D46"]
    assert {key: row["severity"] for key, row in rows.items()} == {
        "D43": "High",
        "D44": "High",
        "D45": "Critical",
        "D46": "High",
    }
    assert all(row["paper_numeric_anchor_eligible"] is False for row in rows.values())
    assert rows["D43"]["value"]["project"][
        "scaled_updates_under_pytorch_rule"
    ] == 0
    assert "persistent PyTorch Adam" in rows["D44"]["topic"]
    assert rows["D45"]["value"]["project"] == {
        "validation_interval_epochs": 40,
        "candidate_count_per_agent_including_epoch_zero": 26,
        "validation_trajectories_per_candidate": 64,
        "selection_metric": "mean_validation_final_state_fidelity",
        "selection_horizon_cycles": 10,
        "selection_ledger_rows": 1040,
    }
    assert rows["D46"]["value"]["project"][
        "warmup_trajectories_population_total"
    ] == 280
    assert rows["D46"]["value"]["project"]["score_baseline_decay"] == 0.95
    assert report["deviation_summary"]["combined_entry_count"] == 46
    assert report["deviation_summary"]["combined_severity_counts"] == {
        "Critical": 24,
        "High": 19,
        "Medium": 3,
        "Low": 0,
    }
    assert report["deviation_summary"]["combined_ids"] == [
        f"D{index:02d}" for index in range(1, 47)
    ]


def test_both_terminal_branches_release_T9_1_4_and_no_go_is_typed_null() -> None:
    report = _published()
    qualified, no_go = report["terminal_state_mapping"]
    assert qualified["terminal_state"] == subject.QUALIFIED_TERMINAL_STATE
    assert no_go["terminal_state"] == subject.NO_GO_TERMINAL_STATE
    assert qualified["releases_t9_1_4"] is True
    assert no_go["releases_t9_1_4"] is True
    assert qualified["matched_phase9_ranking_eligible"] is False
    assert no_go["matched_phase9_ranking_eligible"] is False
    nullable = {
        "parent_report_sha256",
        "numeric_scope",
        "selected_controller",
        "numeric_metrics",
        "rank",
        "official_exact",
        "puviani_surpass",
        "paper_scale_lifetime",
    }
    assert all(no_go["typed_payload"][key] is None for key in nullable)
    simulated_failure = subject._current_terminal_resolution(
        {"status": "FAILED_FAIL_CLOSED"},
        report["terminal_state_mapping"],
    )
    assert (
        simulated_failure["resolved_terminal_state"]
        == subject.NO_GO_TERMINAL_STATE
    )
    assert simulated_failure["releases_t9_1_4"] is True
    assert simulated_failure["typed_payload"] == no_go["typed_payload"]
    with pytest.raises(ValueError, match="not a recognized"):
        subject._current_terminal_resolution(
            {"status": "IN_PROGRESS"},
            report["terminal_state_mapping"],
        )
    assert no_go["match"] == "TERMINAL_NONPASS_WITH_VALID_FAILURE_MANIFEST"
    assert no_go["supported_terminal_states"] == ["FAILED_FAIL_CLOSED"]


def test_claim_slots_and_ranking_boundary_are_closed_even_on_qualified_branch() -> None:
    report = _published()
    assert report["claim_slots"] == {
        "official_exact": None,
        "puviani_surpass": None,
        "paper_scale_lifetime": None,
    }
    assert report["current_terminal_resolution"][
        "resolved_terminal_state"
    ] == subject.QUALIFIED_TERMINAL_STATE
    assert report["current_terminal_resolution"][
        "matched_phase9_ranking_eligible"
    ] is False
    assert report["ranking_boundary"][
        "matched_phase9_ranking_eligible"
    ] is False
    assert report["ranking_boundary"]["sota_claim_eligible"] is False


def test_source_csv_losslessly_carries_all_nested_records() -> None:
    report = _published()
    with SOURCE.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["source_data"]["rows"] == 67
    assert len({(row["section"], row["record_id"]) for row in rows}) == 67
    assert sum(row["section"] == "deviation" for row in rows) == 46
    for row in rows:
        assert (
            hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
            == row["payload_sha256"]
        )
        payload = json.loads(row["payload_json"])
        assert subject._canonical_json(payload) == row["payload_json"]
    decoded = {
        (row["section"], row["record_id"]): json.loads(row["payload_json"])
        for row in rows
    }
    original = _load(
        ROOT / "configs/phase9/t9_1_3_deviation_ledger.json"
    )["deviations"]
    assert decoded[("deviation", "D01")] == original[0]
    assert decoded[("deviation", "D42")] == original[-1]
    assert decoded[("deviation", "D46")] == report["addendum_deviations"][-1]
    assert decoded[("gradient_clipping_reanalysis", "all_40000_updates")] == (
        report["gradient_clipping_reanalysis"]
    )
    assert decoded[
        ("downstream_semantic_seal", subject.SEMANTIC_SEAL_VERSION)
    ] == report["downstream_semantic_seal"]


def test_built_in_mutation_audit_covers_required_attack_surfaces() -> None:
    audit = _published()["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 14
    assert audit["all_detected"] is True
    assert audit["terminal_branch"] == subject.QUALIFIED_BRANCH
    assert audit["branch_case_count"] == 14
    assert all(
        row["mutation_applied"] is True
        and row["terminal_branch"] == subject.QUALIFIED_BRANCH
        and row["rejection_mode"] == "TARGET_GATE_FALSE"
        and row["rejected"] is True
        for row in audit["cases"]
    )
    assert {row["mutation_id"] for row in audit["cases"]} == {
        "fill_official_exact_claim",
        "fill_paper_scale_lifetime_claim",
        "change_terminal_evidence_hash",
        "change_terminal_semantic_evidence",
        "make_no_go_block_T9_1_4",
        "fill_no_go_numeric_payload",
        "promote_to_matched_rank",
        "drift_combined_deviation_count",
        "forge_clipped_update",
        "mutate_branch_deviation_payload",
        "retroactively_reseal_parent",
        "forge_source_row_count",
        "forge_downstream_semantic_seal",
        "forge_mutation_count",
    }


@pytest.mark.parametrize(
    ("target_gate", "mutate"),
    [
        (
            "G09_official_surpass_and_paper_lifetime_claims_stay_null",
            lambda value: value["claim_slots"].update(
                puviani_surpass=1.01
            ),
        ),
        (
            "G09_official_surpass_and_paper_lifetime_claims_stay_null",
            lambda value: value["claim_slots"].update(
                paper_scale_lifetime={"T_ch": 1000.0}
            ),
        ),
        (
            "G03_terminal_evidence_is_branch_local_hash_bound_and_live",
            lambda value: value["parent_bindings"]["parent_report"].update(
                sha256="f" * 64
            ),
        ),
        (
            "G10_qualified_and_typed_no_go_both_release_T9_1_4",
            lambda value: value["terminal_state_mapping"][1].update(
                releases_t9_1_4=False
            ),
        ),
        (
            "G10_qualified_and_typed_no_go_both_release_T9_1_4",
            lambda value: value["terminal_state_mapping"][1][
                "typed_payload"
            ].update(rank=1),
        ),
        (
            "G12_registry_context_cannot_be_promoted_to_matched_or_SOTA_rank",
            lambda value: value["ranking_boundary"].update(
                sota_claim_eligible=True
            ),
        ),
        (
            "G08_deviation_census_is_exact_or_typed_null_by_branch",
            lambda value: value["deviation_summary"][
                "combined_severity_counts"
            ].update(High=18),
        ),
        (
            "G06_gradient_evidence_is_complete_or_typed_null_by_branch",
            lambda value: value["gradient_clipping_reanalysis"]["scopes"][
                "all"
            ].update(maximum=10.1),
        ),
        (
            "G05_addendum_is_nonretroactive_and_parent_reseal_is_prohibited",
            lambda value: value["governance_policy"].update(
                parent_deviation_ledger_edit=True
            ),
        ),
        (
            "G15_targeted_semantic_mutations_all_fail_closed",
            lambda value: value["semantic_mutation_audit"]["cases"][
                0
            ].update(
                rejection_mode=(
                    "EXPECTED_VALIDATION_REJECTION:ValueError"
                )
            ),
        ),
    ],
)
def test_semantic_negative_paths_fail_their_target_gate(
    target_gate: str, mutate: object
) -> None:
    candidate = deepcopy(_published())
    mutate(candidate)
    gates = subject.evaluate_gates(candidate, verify_live_files=False)
    assert gates[target_gate] is False


def test_full_no_go_fixture_build_validate_cli_and_handoff_are_executable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pass_paths = {
        (ROOT / value["path"]).resolve()
        for value in subject.EXPECTED_PARENT_ARTIFACTS.values()
    }
    original_load_json = subject._load_json
    loaded_paths: list[Path] = []

    def guarded_load_json(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        if resolved in pass_paths:
            raise AssertionError(
                f"NO_GO branch touched QUALIFIED artifact: {resolved}"
            )
        loaded_paths.append(resolved)
        return original_load_json(path)

    def forbidden_gradient(*args: object, **kwargs: object) -> object:
        raise AssertionError("NO_GO branch attempted PASS gradient replay")

    monkeypatch.setattr(subject, "_load_json", guarded_load_json)
    monkeypatch.setattr(
        subject, "_gradient_clipping_reanalysis", forbidden_gradient
    )
    with tempfile.TemporaryDirectory(
        prefix="_test_t9_1_3_no_go_", dir=ROOT / "runs"
    ) as directory:
        root = Path(directory)
        manifest_path = _write_live_no_go_fixture(root)
        report_path = root / "no_go_addendum.json"
        source_path = root / "no_go_source.csv"
        assert (
            subject.main(
                [
                    "--report",
                    str(report_path),
                    "--source-data",
                    str(source_path),
                    "--terminal-branch",
                    subject.NO_GO_BRANCH,
                    "--failure-manifest",
                    str(manifest_path),
                ]
            )
            == 0
        )
        cli_payload = json.loads(capsys.readouterr().out)
        assert cli_payload["terminal_branch"] == subject.NO_GO_BRANCH
        report = guarded_load_json(report_path)
        checks = subject.validate_report(
            report,
            verify_live_files=True,
        )
        assert all(checks.values())
        assert report["gate_summary"] == {
            "passed": 16,
            "total": 16,
            "failed": [],
        }
        assert report["semantic_mutation_audit"]["count"] == 14
        assert report["semantic_mutation_audit"]["detected"] == 14
        assert (
            report["semantic_mutation_audit"]["terminal_branch"]
            == subject.NO_GO_BRANCH
        )
        assert all(
            row["terminal_branch"] == subject.NO_GO_BRANCH
            and row["mutation_applied"] is True
            and row["rejection_mode"] == "TARGET_GATE_FALSE"
            and row["rejected"] is True
            for row in report["semantic_mutation_audit"]["cases"]
        )
        assert set(report["parent_bindings"]) == {
            "archive",
            "archive_seal",
            "inventory",
        }
        assert report["parent_snapshot"]["valid_parent_pass_seal"] is False
        assert report["failure_manifest"]["reason"]["code"] == (
            "FINALIZATION_FAILED_NO_VALID_SEAL"
        )
        assert report["gradient_clipping_reanalysis"]["value"] is None
        assert report["gradient_clipping_reanalysis"]["scopes"] is None
        assert report["addendum_deviations"] == []
        assert report["deviation_summary"]["combined_entry_count"] is None
        assert report["claim_slots"] == subject.CLAIM_NULLS
        assert report["current_terminal_resolution"][
            "resolved_terminal_state"
        ] == subject.NO_GO_TERMINAL_STATE
        assert report["current_terminal_resolution"][
            "releases_t9_1_4"
        ] is True
        assert report["downstream_semantic_seal"] == (
            subject._downstream_semantic_seal(report)
        )
        assert report["downstream_semantic_seal"]["consumer_rule"] == (
            subject.DOWNSTREAM_CONSUMER_RULE
        )
        assert report["downstream_semantic_seal"][
            "raw_addendum_hash_required_by_T9_1_4"
        ] is False
        assert source_path.is_file()
        assert (
            subject.main(
                [
                    "--verify",
                    "--report",
                    str(report_path),
                    "--input-contract",
                    str(CONTRACT),
                ]
            )
            == 0
        )
        verify_payload = json.loads(capsys.readouterr().out)
        assert verify_payload["verdict"] == subject.VERDICT_PASS
        assert all(verify_payload["checks"].values())
        archive_path = Path(
            report["failure_manifest"]["failure_evidence"]["archive"][
                "path"
            ]
        )
        with archive_path.open("ab") as stream:
            stream.write(b"live-evidence-drift")
        with pytest.raises(ValueError):
            subject.validate_downstream_handoff(
                report_path,
                input_contract_path=CONTRACT,
            )
    assert not (set(loaded_paths) & pass_paths)


@pytest.mark.parametrize("schema_mutation", ["missing", "extra"])
def test_report_structural_anomalies_raise_explicit_errors(
    schema_mutation: str,
) -> None:
    candidate = deepcopy(_published())
    if schema_mutation == "missing":
        candidate.pop("claim_slots")
    else:
        candidate["unexpected_field"] = None
    with pytest.raises(
        ValueError, match="post-outcome report schema drifted"
    ):
        subject.validate_report(candidate, verify_live_files=False)


def test_failure_manifest_structural_anomaly_raises_explicit_error() -> None:
    with tempfile.TemporaryDirectory(
        prefix="_test_t9_1_3_no_go_schema_", dir=ROOT / "runs"
    ) as directory:
        manifest = _load(
            _write_live_no_go_fixture(Path(directory))
        )
        manifest["unexpected_field"] = None
        with pytest.raises(ValueError, match="schema drifted"):
            subject._validate_failure_manifest_structure(manifest)


@pytest.mark.parametrize(
    "tamper_target",
    [
        "archive",
        "archive_seal",
        "inventory",
        "typed_null",
        "semantic_sha256",
    ],
)
def test_no_go_failure_manifest_tampering_fails_closed(
    tamper_target: str,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="_test_t9_1_3_no_go_tamper_", dir=ROOT / "runs"
    ) as directory:
        manifest_path = _write_live_no_go_fixture(Path(directory))
        manifest = _load(manifest_path)
        if tamper_target in {"archive", "archive_seal", "inventory"}:
            evidence_path = Path(
                manifest["failure_evidence"][tamper_target]["path"]
            )
            with evidence_path.open("ab") as stream:
                stream.write(b"tampered")
        elif tamper_target == "typed_null":
            manifest["typed_null_payload"]["numeric_metrics"] = {
                "lifetime": 1.0
            }
            manifest["semantic_sha256"] = (
                subject._failure_manifest_semantic_sha256(manifest)
            )
        else:
            manifest["semantic_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            subject._validate_failure_manifest_live(
                manifest,
                verify_live_files=True,
            )


def test_generator_is_byte_deterministic_and_never_changes_parent_inputs() -> None:
    parent_paths = [
        ROOT / value["path"]
        for value in subject.EXPECTED_PARENT_ARTIFACTS.values()
    ]
    before = {path: _sha256(path) for path in parent_paths}
    with tempfile.TemporaryDirectory(
        prefix="_test_t9_1_3_addendum_", dir=ROOT / "runs"
    ) as directory:
        source = Path(directory) / "source.csv"
        first = subject.build_report(source_data_path=source)
        first_source = source.read_bytes()
        second = subject.build_report(source_data_path=source)
        second_source = source.read_bytes()
    assert first == second
    assert first_source == second_source
    assert first["analysis_sha256"] == second["analysis_sha256"]
    assert {path: _sha256(path) for path in parent_paths} == before


def test_verify_cli_entry_point_is_read_only_and_passes(capsys: pytest.CaptureFixture[str]) -> None:
    report_before = REPORT.read_bytes()
    source_before = SOURCE.read_bytes()
    assert subject.main(["--verify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == subject.VERDICT_PASS
    assert all(payload["checks"].values())
    assert REPORT.read_bytes() == report_before
    assert SOURCE.read_bytes() == source_before
