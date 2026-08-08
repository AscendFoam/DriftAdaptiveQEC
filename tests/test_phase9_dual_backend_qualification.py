from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_dual_backend_qualification as runner


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
    prereg["formal_matrix"]["samples_per_representative_probe_backend"] = 1
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
    convergence["samples_per_state_action_backend_cutoff"] = 1
    convergence["trajectories_per_fault_backend_cutoff"] = 1
    convergence["additional_cutoff_12_backend_rounds"] = 162
    convergence["total_unique_formal_backend_rounds_after_amendment"] = 694
    return prereg, amendment


@pytest.fixture(scope="module")
def development_evidence() -> runner.EvidenceAccumulator:
    prereg, amendment = _development_contract()
    return runner.execute_matrix(prereg, amendment)


def test_frozen_formal_row_accounting_is_exact() -> None:
    assert runner.expected_row_count(PREREG, AMENDMENT) == 16_800
    matrix = PREREG["formal_matrix"]
    assert (
        5 * 7 * matrix["samples_per_shared_state_action_backend"] * 2
        + 16 * matrix["samples_per_representative_probe_backend"] * 2
        + 6 * 7 * matrix["samples_per_logical_state_action_backend"] * 2
        + 4 * matrix["trajectories_per_fault_backend"] * 12 * 2
        + AMENDMENT["cutoff_convergence_submatrix"][
            "additional_cutoff_12_backend_rounds"
        ]
        == 16_800
    )


def test_development_matrix_uses_no_formal_seed_interval(
    development_evidence: runner.EvidenceAccumulator,
) -> None:
    assert len(development_evidence.rows) == 694
    assert all(int(row["seed"]) >= 1_100_000 for row in development_evidence.rows)
    assert not {
        int(row["seed"]) for row in development_evidence.rows
    }.intersection(range(920_000, 950_048))


def test_development_matrix_has_complete_unique_conservative_ledger(
    development_evidence: runner.EvidenceAccumulator,
) -> None:
    rows = development_evidence.rows
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert not [row for row in rows if row["exception_type"]]
    assert all(row["conservation_pass"] is True for row in rows)
    assert {
        row["probe_id"]
        for row in rows
        if row["layer"] == "probe"
    } == set(PREREG["action_contract"]["probe_ids"])
    assert {
        row["scenario"]
        for row in rows
        if row["layer"] == "fault"
    } == {"step", "telegraph", "burst", "compound"}


def test_development_matrix_keeps_backend_rng_and_seeds_disjoint(
    development_evidence: runner.EvidenceAccumulator,
) -> None:
    rows = development_evidence.rows
    seeds_a = {int(row["seed"]) for row in rows if row["backend"] == "A"}
    seeds_b = {int(row["seed"]) for row in rows if row["backend"] == "B"}
    assert seeds_a.isdisjoint(seeds_b)
    assert {
        row["rng_namespace"] for row in rows if row["backend"] == "A"
    } == {"NUMPY_SEEDSEQUENCE_ADDRESSED"}
    assert {
        row["rng_namespace"] for row in rows if row["backend"] == "B"
    } == {"BLAKE2B_ADDRESS_PYTHON_RANDOM_BOX_MULLER"}


def test_logical_rows_store_trace_decreasing_blocks_without_postselection(
    development_evidence: runner.EvidenceAccumulator,
) -> None:
    logical_rows = [
        row for row in development_evidence.rows if row["layer"] == "logical"
    ]
    assert logical_rows
    for row in logical_rows:
        block = np.array(
            [
                [
                    complex(
                        row["logical_block_00_real"],
                        row["logical_block_00_imag"],
                    ),
                    complex(
                        row["logical_block_01_real"],
                        row["logical_block_01_imag"],
                    ),
                ],
                [
                    complex(
                        row["logical_block_10_real"],
                        row["logical_block_10_imag"],
                    ),
                    complex(
                        row["logical_block_11_real"],
                        row["logical_block_11_imag"],
                    ),
                ],
            ]
        )
        assert np.trace(block).real == pytest.approx(
            row["logical_survival"],
            abs=2.0e-10,
        )
        assert 0.0 <= row["logical_survival"] <= 1.0


def test_cutoff_confirmation_reuses_only_declared_positions(
    development_evidence: runner.EvidenceAccumulator,
) -> None:
    rows = development_evidence.rows
    confirmation = [row for row in rows if row["cutoff"] == 12]
    assert confirmation
    assert all(row["convergence_member"] is True for row in confirmation)
    base_members = [
        row
        for row in rows
        if row["cutoff"] == 8 and row["convergence_member"] is True
    ]
    assert base_members
    for layer in ("shared", "logical", "fault"):
        for backend in ("A", "B"):
            assert {
                row["seed_position"]
                for row in confirmation
                if row["layer"] == layer and row["backend"] == backend
            } == {
                row["seed_position"]
                for row in base_members
                if row["layer"] == layer and row["backend"] == backend
            }


def test_fault_rows_preserve_complete_horizon_and_no_early_exit(
    development_evidence: runner.EvidenceAccumulator,
) -> None:
    trajectories: dict[str, list[dict]] = {}
    for row in development_evidence.rows:
        if row["layer"] == "fault":
            trajectories.setdefault(str(row["trajectory_id"]), []).append(row)
    assert trajectories
    for values in trajectories.values():
        assert len(values) == 12
        assert sorted(int(row["round_index"]) for row in values) == list(
            range(12)
        )
        assert sum(row["terminal_round"] is True for row in values) == 1


def test_formal_runner_refuses_missing_child_seal() -> None:
    with pytest.raises(RuntimeError, match="child seal is missing"):
        runner.verify_child_seal(ROOT / "runs" / "definitely_missing_t924_root")


def test_fault_intervention_semantics_are_exact() -> None:
    scenarios = PREREG["formal_matrix"]["fault_scenarios"]
    step = [
        runner._fault_delta_for_round("step", scenarios["step"], index)
        for index in range(12)
    ]
    assert np.count_nonzero([np.linalg.norm(value) for value in step]) == 1
    assert np.allclose(step[4], scenarios["step"]["drift_delta"])
    burst = [
        runner._fault_delta_for_round("burst", scenarios["burst"], index)
        for index in range(12)
    ]
    assert np.allclose(burst[4], scenarios["burst"]["drift_delta"])
    assert np.allclose(burst[7], -np.asarray(scenarios["burst"]["drift_delta"]))
