from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import pytest

from cnn_fpga.benchmark import phase9_fresh_twin_preformal_audit as audit


ROOT = Path(__file__).resolve().parents[1]


def _passing_snapshot() -> dict[str, object]:
    return {
        "task_id": audit.TASK_ID,
        "lineage_task_id": audit.TASK_ID,
        "lineage_verdict": "PASS_HISTORICAL_NO_GO_LINEAGE_BOUND",
        "lineage_analysis_valid": True,
        "lineage_fresh_scan_live": True,
        "historical_parent_rewritten": False,
        "historical_formal_verdict": "NO_GO_TWIN_QUALIFICATION",
        "fresh_old_cell_reference_count": 0,
        "diagnostic_verdict": "PASS_FRESH_IQ_SEMANTICS_DIAGNOSTIC",
        "diagnostic_analysis_valid": True,
        "diagnostic_old_cell_accessed": False,
        "diagnostic_old_outcome_selected_margin": False,
        "readout_verdict": "PASS_FRESH_READOUT_EMPIRICAL_POWER",
        "readout_analysis_valid": True,
        "readout_old_cell_accessed": False,
        "readout_formal_pool_accessed": False,
        "design_verdict": "PASS_FRESH_TWIN_DESIGN_POWER",
        "design_analysis_valid": True,
        "design_old_cell_accessed": False,
        "design_old_outcome_used": False,
        "design_formal_pool_accessed": False,
        "config_hash_matches_design": True,
        "config_hash_matches_readout": True,
        "qualification_binds_design_report": True,
        "runtime_dependency_contract_exact": True,
        "runtime_dependency_binding_count": 11,
        "gate_blueprint_count": 1589,
        "gate_blueprint_ids_unique": True,
        "gate_blueprint_rows_complete": True,
        "gate_blueprint_directions_exact": True,
        "gate_blueprint_normalized_sd_valid": True,
        "gate_blueprint_declared_row_count": 1589,
        "gate_blueprint_source_config": (
            "configs/phase9/t_risk_20260726_01_design_power.json"
        ),
        "verification_contract_exact": True,
        "gate_blueprint_sha256": "a" * 64,
        "gate_blueprint_declared_sha256": "a" * 64,
        "gate_blueprint_design_sha256": "b" * 64,
        "gate_blueprint_declared_design_sha256": "b" * 64,
        "all_source_csv_nonempty_rectangular": True,
        "all_source_csv_rows_unique": True,
        "all_source_csv_no_nonfinite_tokens": True,
        "split_count": 16,
        "splits_disjoint": True,
        "fresh_seed_namespace": True,
        "formal_round_pool": (
            (1_070_000, 1_070_768),
            (1_071_000, 1_071_768),
        ),
        "formal_trajectory_pool": (
            (1_072_000, 1_072_256),
            (1_073_000, 1_073_256),
        ),
        "selected_round_count": 768,
        "selected_trajectory_count": 256,
        "candidate_round_counts": [128, 256, 512, 768],
        "candidate_trajectory_counts": [96, 192, 256],
        "pilot_allowed_fields": [
            "round_sample_count",
            "trajectory_sample_count",
        ],
        "old_outcome_policy": {
            "historical_no_go_rewritten": False,
            "historical_formal_cell_data_access_allowed": False,
            "old_outcome_may_choose_margin": False,
            "old_outcome_may_choose_endpoint": False,
            "old_outcome_may_choose_family": False,
            "pilot_may_choose": [
                "round_sample_count",
                "trajectory_sample_count",
            ],
        },
        "family_count": 7,
        "family_names": [
            "cutoff_mapping",
            "fault_trajectory_tail",
            "iq_conditional_distribution",
            "likelihood_score_posterior",
            "logical_ptm_survival",
            "physical_state_channel",
            "reset_leakage",
        ],
        "all_family_metrics_have_margin": True,
        "margin_count": 27,
        "all_margins_positive_finite_with_source": True,
        "all_cells_required": True,
        "no_postselection": True,
        "fault_logical_initialization_exact": True,
        "cutoff_ladder": [8, 12, 16, 20],
        "primary_cutoff_increments": [[12, 16], [16, 20]],
        "global_test": "intersection_union_equivalence",
        "cell_test": "two_one_sided_tests",
        "cell_confidence_interval": 0.90,
        "global_type_i_error": 0.05,
        "mixed_unit_vector_max": False,
        "raw_log_evidence_policy": (
            "diagnostic only; never a cross-gain primary gate"
        ),
        "cell_deletion": False,
        "mean_only_rescue": False,
        "nonfinite_policy": audit.INCOMPLETE_FORMAL_VERDICT,
        "cluster_unit": (
            "independent seed position; all rows sharing a seed remain together"
        ),
        "power_pseudoexperiments": 4000,
        "power_formal_pool_access": False,
        "power_false_pass_ucb": 0.05,
        "power_same_a_lcb": 0.90,
        "power_same_b_covered": True,
        "power_ab_lcb": 0.90,
        "power_mutants_covered": True,
        "readout_base_measure": (
            "two_dimensional_lebesgue_per_complex_iq_sample"
        ),
        "sigma_convention": "per_real_axis_standard_deviation",
        "integration_convention": "arithmetic_mean_over_window",
        "latent_conditioning": "one_ancilla_label_per_complete_window",
        "gain_jacobian_rule": "-N*log(abs(det(G))) applied exactly once",
        "proper_score_unit": "nats_per_complex_sample",
        "reference_no_backend_import": True,
        "reference_no_random_or_numpy_import": True,
        "reference_formula_tokens": True,
        "runner_imports_backend_a": True,
        "runner_imports_backend_b": True,
        "runner_no_cli_formal_overrides": True,
        "runner_receipt_exact_tokens": True,
        "runner_complete_attempt_tokens": True,
        "runner_fault_logical_tokens": True,
        "verifier_no_physics_import": True,
        "verifier_three_verdict_tokens": True,
        "verifier_claim_tokens": True,
        "verifier_release_tokens": True,
        "verifier_iut_tokens": True,
        "verifier_no_outcome_hardcode": True,
        "verifier_seal_policy_tokens": True,
        "verifier_terminal_logical_survival_token": True,
        "auditor_no_runtime_import": True,
        "auditor_outcome_blind_fixture_tokens": True,
        "formal_existing": [],
        "commit_sha": "1" * 40,
        "commit_head_exact": True,
        "commit_paths_exact": True,
        "protected_path_count": 24,
        "fixture_pass": True,
        "fixture_no_go": True,
        "fixture_incomplete": True,
    }


def test_snapshot_contract_has_seventy_independent_gates() -> None:
    gates = audit.audit_snapshot(_passing_snapshot())
    assert len(gates) == 70
    assert all(gates.values()), {
        name: passed for name, passed in gates.items() if not passed
    }


def test_every_semantic_mutation_is_killed() -> None:
    mutations = audit.run_mutation_audit(_passing_snapshot())
    assert len(mutations) == 72
    assert len({row["mutation_id"] for row in mutations}) == 72
    assert all(row["detected"] is True for row in mutations)
    assert all(row["failed_gates"] for row in mutations)


@pytest.mark.parametrize("branch", ["pass", "no_go", "incomplete"])
def test_all_three_outcome_blind_fixtures_pass(branch: str) -> None:
    fixture = audit._formal_fixture(branch)
    result = audit.audit_outcome_fixture(fixture)
    assert len(result) == 6
    assert all(result.values())
    assert tuple(fixture["claim_state"]) == audit.CLAIM_FIELDS
    assert all(value is None for value in fixture["claim_state"].values())


def test_outcome_fixture_mutations_are_all_killed() -> None:
    mutations = audit._fixture_mutations()
    assert len(mutations) == 18
    assert all(row["detected"] is True for row in mutations)


@pytest.mark.parametrize(
    ("branch", "field", "replacement"),
    [
        ("pass", "qualified_claim", None),
        ("pass", "released_tasks", []),
        ("pass", "infrastructure_errors", ["NaN"]),
        ("no_go", "qualified_claim", audit.QUALIFIED_CLAIM),
        ("no_go", "released_tasks", list(audit.DOWNSTREAM_TASKS)),
        ("no_go", "blocked_tasks", []),
        ("incomplete", "infrastructure_errors", []),
        ("incomplete", "complete_finite_denominator", True),
        ("incomplete", "released_tasks", ["T9.2.5"]),
    ],
)
def test_branch_swap_claim_and_release_mutants_fail(
    branch: str,
    field: str,
    replacement: object,
) -> None:
    fixture = audit._formal_fixture(branch)
    fixture[field] = replacement
    assert not all(audit.audit_outcome_fixture(fixture).values())


def test_claim_schema_rejects_bool_alias_missing_extra_and_wrong_order() -> None:
    baseline = audit._formal_fixture("pass")
    mutations = []
    bool_alias = copy.deepcopy(baseline)
    bool_alias["claim_state"]["rank"] = False
    mutations.append(bool_alias)
    missing = copy.deepcopy(baseline)
    missing["claim_state"].pop("rank")
    mutations.append(missing)
    extra = copy.deepcopy(baseline)
    extra["claim_state"]["surpass"] = None
    mutations.append(extra)
    wrong_order = copy.deepcopy(baseline)
    wrong_order["claim_state"] = dict(reversed(wrong_order["claim_state"].items()))
    mutations.append(wrong_order)
    assert all(
        not audit.audit_outcome_fixture(value)[
            "F02_exact_fifteen_typed_null"
        ]
        for value in mutations
    )


def test_auditor_imports_no_physics_runner_or_verifier() -> None:
    source = (
        ROOT
        / "cnn_fpga/benchmark/phase9_fresh_twin_preformal_audit.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith("physics") for name in imports)
    assert not any(
        "phase9_fresh_twin_qualification" in name for name in imports
    )
    assert not any("phase9_fresh_twin_verifier" in name for name in imports)


def test_auditor_has_no_historical_cell_level_literal() -> None:
    source = (
        ROOT
        / "cnn_fpga/benchmark/phase9_fresh_twin_preformal_audit.py"
    ).read_text(encoding="utf-8")
    prohibited = (
        "t9_2_4_dual_backend_" + "cell_ledger.csv",
        "t9_2_4_dual_backend_" + "qualification_source_data.csv",
        "t9_2_4_dual_backend_" + "state_archive.npz",
    )
    assert not any(value in source for value in prohibited)


def test_fresh_lineage_scan_rejects_missing_source(tmp_path: Path) -> None:
    source = (
        tmp_path
        / "cnn_fpga/benchmark/phase9_fresh_twin_example.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("# fresh\n", encoding="utf-8")
    complete = {
        "fresh_source_scan": {
            "scanned_paths": [
                "cnn_fpga/benchmark/phase9_fresh_twin_example.py"
            ],
            "violations": [],
        }
    }
    assert audit._fresh_lineage_scan_live(tmp_path, complete)
    complete["fresh_source_scan"]["scanned_paths"] = []
    assert not audit._fresh_lineage_scan_live(tmp_path, complete)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_commit_binding_requires_exact_head_and_committed_bytes(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "phase9@example.invalid")
    _git(tmp_path, "config", "user.name", "Phase9 Test")
    protected = tmp_path / "bound.txt"
    protected.write_text("frozen\n", encoding="utf-8", newline="\n")
    _git(tmp_path, "add", "bound.txt")
    _git(tmp_path, "commit", "-m", "preformal")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    receipt = audit._commit_binding(tmp_path, commit, ["bound.txt"])
    assert receipt["all_protected_paths_exact"] is True
    assert receipt["head_sha"] == commit

    protected.write_text("mutated\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="differ from commit"):
        audit._commit_binding(tmp_path, commit, ["bound.txt"])


def test_commit_binding_rejects_short_or_non_head_commit(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "phase9@example.invalid")
    _git(tmp_path, "config", "user.name", "Phase9 Test")
    protected = tmp_path / "bound.txt"
    protected.write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", "bound.txt")
    _git(tmp_path, "commit", "-m", "one")
    first = _git(tmp_path, "rev-parse", "HEAD")
    protected.write_text("two\n", encoding="utf-8")
    _git(tmp_path, "add", "bound.txt")
    _git(tmp_path, "commit", "-m", "two")
    with pytest.raises(ValueError, match="exact lowercase 40-hex"):
        audit._commit_binding(tmp_path, first[:12], ["bound.txt"])
    with pytest.raises(ValueError, match="exact preformal HEAD"):
        audit._commit_binding(tmp_path, first, ["bound.txt"])


def test_analysis_hash_validator_rejects_nan_and_tamper() -> None:
    value = {"task_id": audit.TASK_ID, "verdict": "PASS"}
    value["analysis_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert audit._analysis_hash_valid(value)
    value["verdict"] = "NO_GO"
    assert not audit._analysis_hash_valid(value)
    value["analysis_sha256"] = "0" * 64
    assert not audit._analysis_hash_valid(value)


def test_csv_audit_rejects_duplicate_nan_and_ragged_rows(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean.csv"
    clean.write_text("id,value\n1,0.2\n2,0.3\n", encoding="utf-8")
    assert audit._csv_shape(clean) == (2, 2, True, True, True)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("id,value\n1,0.2\n1,0.2\n", encoding="utf-8")
    assert audit._csv_shape(duplicate) == (2, 2, True, False, True)

    nonfinite = tmp_path / "nonfinite.csv"
    nonfinite.write_text("id,value\n1,NaN\n", encoding="utf-8")
    assert audit._csv_shape(nonfinite) == (1, 2, True, True, False)

    ragged = tmp_path / "ragged.csv"
    ragged.write_text("id,value\n1\n", encoding="utf-8")
    assert audit._csv_shape(ragged) == (1, 2, False, True, True)


def test_source_binding_assembler_cannot_drop_runtime_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = {"formal_runner": "runner.py", "formal_verifier": "verifier.py"}
    runtime = ("runtime/a.py", "runtime/b.py")
    monkeypatch.setattr(audit, "SOURCE_PATHS", direct)
    monkeypatch.setattr(audit, "RUNTIME_DEPENDENCY_PATHS", runtime)
    for relative in (*direct.values(), *runtime):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8", newline="\n")

    bindings = audit._all_source_bindings(tmp_path)
    assert set(bindings) == {
        "formal_runner",
        "formal_verifier",
        "runtime_dependency_00",
        "runtime_dependency_01",
    }
    assert bindings["runtime_dependency_00"]["path"] == runtime[0]
    assert bindings["runtime_dependency_01"]["path"] == runtime[1]

    # The persisted audit is the source of the production seal.  This guards
    # against reintroducing a second, direct-only assembler there.
    assert "_all_source_bindings(root)" in inspect.getsource(audit.build_audit)


def test_verify_seal_checks_self_hash_live_binding_and_one_shot_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = tmp_path / "bound.json"
    bound.write_text("{}\n", encoding="utf-8", newline="\n")
    binding = audit._binding(tmp_path, "bound.json")
    preaudit = tmp_path / "preaudit.json"
    preaudit_document: dict[str, object] = {
        "schema_version": audit.AUDIT_SCHEMA_VERSION,
        "verdict": audit.PASS_VERDICT,
    }
    preaudit_document["analysis_sha256"] = audit._sha(
        preaudit_document
    )
    preaudit.write_text(
        json.dumps(preaudit_document) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    preaudit_binding = audit._binding(tmp_path, "preaudit.json")
    runtime_bindings: dict[str, dict[str, object]] = {}
    for index, relative in enumerate(audit.RUNTIME_DEPENDENCY_PATHS):
        runtime_path = tmp_path / relative
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            f"# runtime {index}\n",
            encoding="utf-8",
            newline="\n",
        )
        runtime_bindings[f"runtime_dependency_{index:02d}"] = (
            audit._binding(tmp_path, relative)
        )
    seal: dict[str, object] = {
        "schema_version": audit.SCHEMA_VERSION,
        "verdict": audit.PASS_VERDICT,
        "preformal_audit_analysis_sha256": preaudit_document[
            "analysis_sha256"
        ],
        "live_bindings": {
            "fresh_config": binding,
            "fresh_runner": binding,
            "historical_lineage_receipt": binding,
            "design_power_report": binding,
            "preformal_audit": preaudit_binding,
            **runtime_bindings,
        },
        "input_bindings": {"x": binding},
        "source_bindings": {"x": binding},
        "test_bindings": {"x": binding},
    }
    seal["analysis_sha256"] = audit._sha(seal)
    monkeypatch.setattr(audit, "FORMAL_OUTPUT_PATHS", ("formal.json",))
    assert audit.verify_seal(seal, tmp_path)

    formal = tmp_path / "formal.json"
    formal.write_text("{}\n", encoding="utf-8")
    assert not audit.verify_seal(seal, tmp_path)
    assert audit.verify_seal(
        seal, tmp_path, require_outputs_absent=False
    )

    bound.write_text('{"tampered":true}\n', encoding="utf-8")
    assert not audit.verify_seal(
        seal, tmp_path, require_outputs_absent=False
    )


def test_write_seal_is_strict_json_and_preserves_nulls(
    tmp_path: Path,
) -> None:
    value = {
        "schema_version": audit.SCHEMA_VERSION,
        "claim_state": {field: None for field in audit.CLAIM_FIELDS},
    }
    value["analysis_sha256"] = audit._sha(value)
    output = tmp_path / "seal.json"
    audit.write_seal(output, value)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == value
    assert all(item is None for item in loaded["claim_state"].values())
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="one-shot"):
        audit.write_seal(output, value)

    audit_output = tmp_path / "audit.json"
    audit.write_audit(audit_output, value)
    assert json.loads(audit_output.read_text(encoding="utf-8")) == value
    with pytest.raises(FileExistsError, match="one-shot"):
        audit.write_audit(audit_output, value)


def test_formal_output_registry_contains_attempt_archive_and_verifier() -> None:
    names = set(audit.FORMAL_OUTPUT_PATHS)
    assert "docs/t_risk_20260726_01_fresh_attempt_ledger.jsonl" in names
    assert "docs/t_risk_20260726_01_fresh_cell_ledger.csv" in names
    assert "docs/t_risk_20260726_01_fresh_raw_archive.zip" in names
    assert "docs/t_risk_20260726_01_fresh_verification.json" in names
    assert "docs/t_risk_20260726_01_fresh_release.json" in names


def test_runner_live_binding_schema_is_exact() -> None:
    required = {
        "fresh_config",
        "fresh_runner",
        "historical_lineage_receipt",
        "design_power_report",
        "preformal_audit",
    }
    # Guard the exact externally consumed field names without importing runner.
    source = (
        ROOT
        / "cnn_fpga/benchmark/phase9_fresh_twin_preformal_audit.py"
    ).read_text(encoding="utf-8")
    assert all(f'"{name}"' in source for name in required)
    assert audit.STATUS == "PASS_FRESH_TWIN_PREFORMAL_AUDIT_SEALED"
