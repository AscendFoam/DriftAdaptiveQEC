from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.trajectory_lookup_control_oracle import (
    DEFAULT_ARTIFACT,
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_DATA,
    PAPER_SOURCE,
    implementation_sha256,
)
from physics.trajectory_lookup_control_oracle import (
    TrajectoryLookupConfig,
    evaluate_exact_policy,
    load_policy_from_state,
)

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: object) -> str:
    array = torch.as_tensor(value).detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ROOT / DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checkpoint() -> dict:
    try:
        return torch.load(ROOT / DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ROOT / DEFAULT_CHECKPOINT, map_location="cpu")


def test_artifact_passes_all_live_gates(artifact: dict) -> None:
    assert artifact["task_id"] == "T3.2.9"
    assert artifact["status"] == "PASS"
    assert artifact["required_gates"] == list(artifact["gates"])
    assert len(artifact["gates"]) == 20
    assert all(artifact["gates"].values())
    assert artifact["implementation_sha256"] == implementation_sha256()


def test_literature_checkpoint_and_source_hashes_are_live(artifact: dict) -> None:
    assert artifact["literature"]["source_sha256"] == sha(ROOT / PAPER_SOURCE)
    assert artifact["checkpoint"]["sha256"] == sha(ROOT / DEFAULT_CHECKPOINT)
    assert artifact["source_data"]["sha256"] == sha(ROOT / DEFAULT_SOURCE_DATA)


def test_tree_is_causal_and_has_the_exact_two_cycle_size(artifact: dict) -> None:
    tree = artifact["tree_contract"]
    assert tree["terminal_branches"] == 16
    assert tree["causal_history_nodes"] == 15
    assert tree["lookup_action_scalars"] == 225
    assert tree["decision_timing"] == "action at half-cycle j sees exactly outcomes [0,j)"
    covered = []
    for depth in tree["prefix_coverage"]:
        assert depth["unique_indices"] == list(
            range(depth["expected_start"], depth["expected_stop_exclusive"])
        )
        covered.extend(depth["unique_indices"])
    assert covered == list(range(15))


def test_source_data_retains_all_optimization_points_and_branches(artifact: dict) -> None:
    with (ROOT / DEFAULT_SOURCE_DATA).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == artifact["source_data"]["row_count"] == 3418
    assert {row["row_type"] for row in rows} == {
        "optimization_trace",
        "evaluation_branch",
        "resource_growth",
    }
    traces = [row for row in rows if row["row_type"] == "optimization_trace"]
    branches = [row for row in rows if row["row_type"] == "evaluation_branch"]
    resources = [row for row in rows if row["row_type"] == "resource_growth"]
    assert len(traces) == 3312
    assert len(branches) == 96
    assert len(resources) == 10
    for family in ("time_indexed_open_loop", "causal_history_lookup"):
        for phase, expected in (("phase_one", 301), ("refinement", 251)):
            for restart in ("0", "1", "2"):
                subset = [
                    row
                    for row in traces
                    if row["family"] == family
                    and row["phase"] == phase
                    and row["restart_index"] == restart
                ]
                assert len(subset) == expected


def test_all_restarts_are_real_and_selected_refinements_are_converged(artifact: dict) -> None:
    optimization = artifact["optimization"]
    for phase in ("phase_one", "refinement"):
        for family in ("time_indexed_open_loop", "causal_history_lookup"):
            runs = optimization[phase][family]
            assert len(runs) == 3
            assert all(run["gradient_covered_nodes"] == run["action_node_count"] for run in runs)
            assert all(run["changed_nodes"] == run["action_node_count"] for run in runs)
    selected_open = optimization["selected_open_loop_restart"]
    selected_lookup = optimization["selected_lookup_restart"]
    assert optimization["refinement"]["time_indexed_open_loop"][selected_open][
        "last_25_epoch_gain"
    ] <= 2.0e-4
    assert optimization["refinement"]["causal_history_lookup"][selected_lookup][
        "last_25_epoch_gain"
    ] <= 2.0e-4


def test_lookup_retains_nested_open_loop_and_improves_it(artifact: dict) -> None:
    evaluations = artifact["evaluations"]
    standard = evaluations["standard"]["12"]["expected_fidelity"]
    open_loop = evaluations["time_indexed_open_loop"]["12"]["expected_fidelity"]
    lookup = evaluations["lookup"]["12"]["expected_fidelity"]
    assert artifact["optimization"]["warm_start_lookup_fidelity"] == pytest.approx(
        open_loop, abs=2.0e-10
    )
    assert open_loop > standard
    assert lookup > open_loop
    assert artifact["comparisons"]["primary_lookup_minus_standard_fidelity"] == pytest.approx(
        lookup - standard
    )


def test_checkpoint_contains_every_restart_and_live_selected_tables(
    artifact: dict, checkpoint: dict
) -> None:
    assert checkpoint["schema"] == artifact["schema"]
    assert checkpoint["implementation_sha256"] == artifact["implementation_sha256"]
    for family, key, shape in (
        ("time_indexed_open_loop", "open_loop", (4, 15)),
        ("causal_history_lookup", "lookup", (15, 15)),
    ):
        state = checkpoint[key]
        assert tuple(state["raw_table"].shape) == shape
        assert len(state["all_restart_states"]) == 3
        expected_hash = artifact["optimization"][
            f"selected_{'open_loop' if key == 'open_loop' else 'lookup'}_table_sha256"
        ]
        assert tensor_sha(state["raw_table"]) == expected_hash
        assert state["family"] == family


def test_selected_controls_do_not_win_by_pinning_the_bounded_action_map(
    checkpoint: dict,
) -> None:
    for key in ("open_loop", "lookup"):
        bounded_residual_fraction = torch.tanh(checkpoint[key]["raw_table"]).abs()
        assert float(bounded_residual_fraction.max()) < 0.9
        assert int((bounded_residual_fraction > 0.95).sum()) == 0


def test_checkpoint_replay_recomputes_primary_lookup(artifact: dict, checkpoint: dict) -> None:
    config = TrajectoryLookupConfig(**{
        key: value
        for key, value in checkpoint["config"].items()
        if key not in {"physics_scope", "parameter_names", "nominal_controls"}
    })
    policy = load_policy_from_state(config, checkpoint["lookup"])
    replay = evaluate_exact_policy(config, policy, cutoff=config.cutoff)
    expected = artifact["evaluations"]["lookup"][str(config.cutoff)]
    assert replay.expected_fidelity == pytest.approx(expected["expected_fidelity"], abs=2.0e-10)
    assert replay.trajectory_probability_sum == pytest.approx(1.0, abs=2.0e-10)
    assert replay.maximum_trace_error <= 2.0e-10


def test_cutoff_confirmation_is_frozen_transfer_not_selection(artifact: dict) -> None:
    assert artifact["optimization"]["evaluation_not_used_for_selection"] is True
    assert set(artifact["evaluations"]["lookup"]) == {"12", "16"}
    assert artifact["comparisons"]["confirmation_lookup_minus_standard_fidelity"] > 0.0


def test_resource_projection_exposes_exponential_infeasibility(artifact: dict) -> None:
    final = artifact["resource_growth"][-1]
    assert final["full_cycles"] == 10
    assert final["terminal_branches"] == 1_048_576
    assert final["causal_history_nodes"] == 1_048_575
    assert final["lookup_action_scalars"] == 15_728_625
    assert final["complex128_terminal_state_bytes_lower_bound"] > 9_000_000_000


def test_branch_distribution_cost_and_claim_boundary_are_not_hidden(artifact: dict) -> None:
    standard = artifact["evaluations"]["standard"]["12"]
    lookup = artifact["evaluations"]["lookup"]["12"]
    assert lookup["minimum_trajectory_probability"] < 1.0e-4
    assert lookup["maximum_trajectory_probability"] > 0.6
    assert lookup["expected_ground_outcome_fraction"] < standard[
        "expected_ground_outcome_fraction"
    ]
    forbidden = artifact["claim_boundary"]["forbidden"]
    assert "globally certified ansatz optimum" in forbidden
    assert "decoder oracle or channel-recovery bound" in forbidden
    assert "long-horizon deployable lookup controller" in forbidden
