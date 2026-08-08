from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_twin_formal_runner_amendment as seal


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/phase9/t9_2_4_formal_runner_amendment.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_build_child_seal_is_outcome_blind_and_complete() -> None:
    report = seal.build_seal(ROOT, CONFIG)
    assert report["task_id"] == "T9.2.4"
    assert report["schema_version"] == seal.SCHEMA
    assert report["status"] == seal.STATUS
    assert report["formal_result_accessed"] is False
    assert report["parent_preregistration_analysis_sha256"] == (
        seal.EXPECTED_PARENT_ANALYSIS
    )
    assert report["gate_summary"] == {
        "passed": 30,
        "total": 30,
        "all_passed": True,
    }
    assert report["mutation_summary"] == {
        "detected": 30,
        "total": 30,
        "all_detected": True,
    }
    assert report["formal_row_accounting"][
        "total_unique_backend_rounds"
    ] == 16_800
    assert report["mapping_resolution"][
        "physical_cutoff_convergence_claim"
    ] is None
    assert set(report["claim_state"]) == set(seal.EXPECTED_TYPED_NULL)
    assert all(value is None for value in report["claim_state"].values())
    assert len(report["analysis_sha256"]) == 64


def test_child_seal_binds_runner_verifier_bridge_and_all_tests() -> None:
    report = seal.build_seal(ROOT, CONFIG)
    bindings = report["live_bindings"]
    assert {
        "parent_preregistration",
        "parent_config",
        "amendment_config",
        "amendment_generator",
        "mapping_bridge",
        "formal_runner",
        "independent_verifier",
        "bridge_tests",
        "runner_tests",
        "verifier_tests",
        "amendment_tests",
        "lfs_rules",
    } == set(bindings)
    for binding in bindings.values():
        assert (ROOT / binding["path"]).is_file()
        assert binding["bytes"] > 0
        assert len(binding["sha256"]) == 64


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("formal_result_accessed_before_amendment",), True),
        (
            (
                "immutable_parent_tolerances",
                "maximum_code_projector_frobenius",
            ),
            0.31,
        ),
        (
            (
                "cutoff_convergence_submatrix",
                "total_unique_formal_backend_rounds_after_amendment",
            ),
            14_976,
        ),
        (
            ("statistical_procedure", "bootstrap_resamples"),
            1999,
        ),
        (
            ("statistical_procedure", "resampling_unit"),
            "round",
        ),
        (
            ("failure_and_release_policy", "released_after_scientific_fail"),
            ["T9.2.5", "T9.2.6"],
        ),
        (
            ("claim_boundary", "typed_null"),
            ["rank"],
        ),
    ],
)
def test_amendment_semantic_mutations_fail_closed(
    path: tuple[str, ...],
    value: object,
) -> None:
    candidate = copy.deepcopy(CONFIG)
    cursor = candidate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        seal.build_seal(ROOT, candidate)


def test_existing_output_path_is_rejected_before_formal() -> None:
    candidate = copy.deepcopy(CONFIG)
    candidate["artifact_paths"][
        "report"
    ] = "configs/phase9/t9_2_4_twin_qualification.json"
    with pytest.raises(ValueError, match="formal outputs already exist"):
        seal.build_seal(ROOT, candidate)


def test_mutation_audit_targets_every_gate_exactly_once() -> None:
    report = seal.build_seal(ROOT, CONFIG)
    gates = set(report["gates"])
    targets = [
        item["target_gate"]
        for item in report["semantic_mutation_audit"]
    ]
    assert len(targets) == len(set(targets)) == 30
    assert set(targets) == gates
    assert all(
        item["detected"] is True
        for item in report["semantic_mutation_audit"]
    )
