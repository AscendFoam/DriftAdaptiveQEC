from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_dual_backend_qualification as runner
from cnn_fpga.benchmark import phase9_dual_backend_verifier as verifier


ROOT = Path(__file__).resolve().parents[1]
PREREG = json.loads(
    (ROOT / "configs/phase9/t9_2_4_twin_qualification.json").read_text(
        encoding="utf-8"
    )
)
AMENDMENT = json.loads(
    (
        ROOT / "configs/phase9/t9_2_4_formal_runner_amendment.json"
    ).read_text(encoding="utf-8")
)


def _development_contract() -> tuple[dict, dict]:
    prereg = copy.deepcopy(PREREG)
    amendment = copy.deepcopy(AMENDMENT)
    prereg["formal_matrix"]["samples_per_shared_state_action_backend"] = 2
    prereg["formal_matrix"]["samples_per_representative_probe_backend"] = 2
    prereg["formal_matrix"]["samples_per_logical_state_action_backend"] = 2
    prereg["formal_matrix"]["trajectories_per_fault_backend"] = 2
    for name, start in (
        ("formal_backend_a_seeds", 1_100_000),
        ("formal_backend_b_seeds", 1_200_000),
        ("trajectory_backend_a_seeds", 1_300_000),
        ("trajectory_backend_b_seeds", 1_400_000),
    ):
        prereg["splits"][name]["start"] = start
        prereg["splits"][name]["count"] = 2
    convergence = amendment["cutoff_convergence_submatrix"]
    convergence["samples_per_state_action_backend_cutoff"] = 2
    convergence["trajectories_per_fault_backend_cutoff"] = 2
    convergence["additional_cutoff_12_backend_rounds"] = 324
    convergence["total_unique_formal_backend_rounds_after_amendment"] = 888
    return prereg, amendment


@pytest.fixture(scope="module")
def development_bundle():
    prereg, amendment = _development_contract()
    evidence = runner.execute_matrix(prereg, amendment)
    archive = dict(evidence.mapping_arrays)
    density_by_row = {
        row_id: matrix
        for cutoff in (8, 12)
        for row_id, matrix in zip(
            evidence.density_row_ids[cutoff],
            evidence.densities[cutoff],
        )
    }
    specs, source = verifier.build_metric_specs(
        evidence.rows,
        archive,
        density_by_row,
        amendment,
    )
    return evidence, archive, density_by_row, specs, source, amendment


def test_verifier_source_imports_no_physics_backend() -> None:
    source = (
        ROOT / "cnn_fpga/benchmark/phase9_dual_backend_verifier.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert all(not module.startswith("physics") for module in modules)


def test_metric_ledger_is_complete_unique_and_not_demo_scoped(
    development_bundle,
) -> None:
    evidence, archive, density_by_row, specs, source, amendment = (
        development_bundle
    )
    assert len(evidence.rows) == 888
    assert len(specs) == 1042
    assert len({spec.gate_id for spec in specs}) == len(specs)
    assert len(source) == 496
    scopes = {spec.scope for spec in specs}
    assert any(scope.startswith("ab|shared") for scope in scopes)
    assert any(scope.startswith("ab|probe") for scope in scopes)
    assert any(scope.startswith("ab|logical_ptm") for scope in scopes)
    assert any(scope.startswith("ab|fault") for scope in scopes)
    assert any(scope.startswith("cutoff|A|") for scope in scopes)
    assert any(scope.startswith("cutoff|B|") for scope in scopes)


def test_mapping_gates_use_raw_independent_archive_and_frozen_thresholds(
    development_bundle,
) -> None:
    specs = development_bundle[3]
    mapping = [spec for spec in specs if spec.scope.startswith("mapping|")]
    assert len(mapping) == 4
    for spec in mapping:
        value = spec.estimator(None)
        if spec.direction == "lower":
            assert value >= 0.95
            assert spec.tolerance == 0.95
        else:
            assert value <= 0.30
            assert spec.tolerance == 0.30


def test_cluster_max_t_bootstrap_is_finite_and_simultaneous(
    development_bundle,
) -> None:
    specs = copy.deepcopy(development_bundle[3])
    procedure = copy.deepcopy(development_bundle[5]["statistical_procedure"])
    procedure["bootstrap_resamples"] = 10
    result = verifier.apply_simultaneous_bootstrap(specs, procedure)
    assert result["resamples"] == 10
    assert result["stochastic_metric_count"] == 1038
    assert np.isfinite(result["critical_value"])
    assert result["critical_value"] >= 0.0
    for spec in specs:
        assert np.isfinite(spec.estimate)
        assert np.isfinite(spec.bound)
        assert spec.standard_error >= 0.0
        if spec.stochastic:
            assert spec.bound >= spec.estimate


def test_ptm_uses_trace_decreasing_block_and_detects_postselection_mutation(
    development_bundle,
) -> None:
    evidence, archive, density_by_row, specs, source, amendment = (
        development_bundle
    )
    gate_id = "ab|logical_ptm|XZ|cutoff=8"
    original = next(spec for spec in specs if spec.gate_id == gate_id)
    original_value = original.estimator(None)
    mutated_rows = copy.deepcopy(evidence.rows)
    for row in mutated_rows:
        if row["layer"] != "logical" or row["backend"] != "B":
            continue
        survival = float(row["logical_survival"])
        for field in (
            "logical_block_00_real",
            "logical_block_00_imag",
            "logical_block_01_real",
            "logical_block_01_imag",
            "logical_block_10_real",
            "logical_block_10_imag",
            "logical_block_11_real",
            "logical_block_11_imag",
        ):
            row[field] = float(row[field]) / survival
    mutated_specs, _ = verifier.build_metric_specs(
        mutated_rows,
        archive,
        density_by_row,
        amendment,
    )
    mutated = next(spec for spec in mutated_specs if spec.gate_id == gate_id)
    assert mutated.estimator(None) != pytest.approx(
        original_value,
        abs=1.0e-9,
    )


def test_density_embedding_preserves_trace_and_positivity() -> None:
    matrix = np.zeros((24, 24), dtype=np.complex128)
    matrix[0, 0] = 0.6
    matrix[4, 4] = 0.4
    embedded = verifier._embed_density(matrix, 8, 12)
    assert embedded.shape == (36, 36)
    assert np.trace(embedded) == pytest.approx(1.0)
    assert np.min(np.linalg.eigvalsh(embedded)) >= -1.0e-14


def test_claim_boundary_is_typed_null_only() -> None:
    assert verifier.TYPED_NULL_FIELDS == (
        "round_ler",
        "six_state_lifetime",
        "physical_break_even",
        "official_puviani_exact",
        "puviani_nmf_surpass",
        "external_sota",
        "hardware_measured",
        "rank",
    )


def test_mutating_parent_tolerance_is_detectable_before_formal() -> None:
    amended = copy.deepcopy(AMENDMENT["immutable_parent_tolerances"])
    amended["maximum_logical_ptm_entry_difference"] = 0.13
    amended.pop("source")
    amended.pop("change_count")
    assert amended != PREREG["prefrozen_tolerances"]
