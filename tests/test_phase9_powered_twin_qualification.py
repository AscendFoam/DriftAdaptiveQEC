from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_powered_twin_qualification as subject
from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    build_cell_plan,
    heldout_seed,
    load_config,
    plan_payload,
)
from cnn_fpga.benchmark.phase9_powered_twin_qualification import (
    _record_rb_failure,
    execute_cell_to_store,
)


ROOT = Path(__file__).resolve().parents[1]
PRE_DRIFT_FIELDS = tuple(
    f"pre_intervention_drift_{index}" for index in range(5)
)
INPUT_DRIFT_FIELDS = tuple(
    f"input_intervention_drift_{index}" for index in range(5)
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _isolated_paths(tmp_path: Path) -> dict[str, str]:
    return {
        "object_store": _relative(tmp_path / "objects" / "sha256"),
        "staging_directory": _relative(tmp_path / "staging"),
        "receipt_directory": _relative(tmp_path / "receipts"),
    }


def _source_snapshot() -> str:
    return sha256(
        (ROOT / "cnn_fpga/benchmark/phase9_powered_twin_qualification.py")
        .read_bytes()
    ).hexdigest()


def _objects(receipt: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["role"]): item
        for item in receipt["objects"]  # type: ignore[index]
    }


def _read_rows(binding: dict[str, object]) -> list[dict[str, str]]:
    with (ROOT / str(binding["path"])).open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _expected_fault_delta(
    scenario: str,
    specification: dict[str, object],
    round_index: int,
) -> np.ndarray:
    delta = np.asarray(specification["drift_delta"], dtype="<f8")
    expected = np.zeros(5, dtype="<f8")
    if scenario == "step":
        if round_index == int(specification["change_round"]):
            expected += delta
    elif scenario == "telegraph":
        period = int(specification["period"])
        if round_index % period == 0:
            expected += (
                delta
                if (round_index // period) % 2 == 0
                else -delta
            )
    elif scenario == "burst":
        start = int(specification["start_round"])
        if round_index == start:
            expected += delta
        if round_index == start + int(specification["duration"]):
            expected -= delta
    elif scenario == "compound":
        if round_index == int(specification["change_round"]):
            expected += delta
        start = int(specification["burst_start"])
        if round_index == start:
            expected += delta
        if round_index == start + int(specification["burst_duration"]):
            expected -= delta
    else:
        raise AssertionError(f"unexpected test scenario {scenario}")
    return expected


def _independent_application_receipt(row: dict[str, str]) -> str:
    payload = {
        "schema": "PHASE9-INTERVENTION-APPLICATION-RECEIPT-V1",
        "row_id": row["row_id"],
        "scenario": row["scenario"],
        "round_index": int(row["round_index"]),
        "intervention_delta_sha256": row[
            "intervention_delta_sha256"
        ],
        "intervention_applied": row["intervention_applied"] == "True",
        "pre_intervention_drift_hex": [
            float(row[name]).hex() for name in PRE_DRIFT_FIELDS
        ],
        "input_intervention_drift_hex": [
            float(row[name]).hex() for name in INPUT_DRIFT_FIELDS
        ],
        "pre_intervention_non_drift_state_sha256": row[
            "pre_intervention_non_drift_state_sha256"
        ],
        "input_non_drift_state_sha256": row[
            "input_non_drift_state_sha256"
        ],
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _assert_intervention_mutations_rejected(
    row: dict[str, str],
    expected_delta: np.ndarray,
) -> None:
    mutations = {
        "drift": (
            "input_intervention_drift_2",
            str(float(row["input_intervention_drift_2"]) + 0.25),
        ),
        "non_drift": (
            "input_non_drift_state_sha256",
            "0" * 64,
        ),
        "delta": ("intervention_delta_sha256", "0" * 64),
        "receipt": (
            "intervention_application_receipt_sha256",
            "0" * 64,
        ),
    }
    for name, (field, value) in mutations.items():
        corrupted = dict(row)
        corrupted[field] = value
        with pytest.raises(RuntimeError, match="intervention"):
            subject._validate_intervention_application_fields(
                corrupted,
                expected_delta,
            )


def test_execution_config_uses_frozen_t04_fault_scenario_authority() -> None:
    config, _ = load_config(ROOT)
    parent_path = ROOT / str(
        config["parent_evidence"]["fresh_twin_parent_config"]["path"]
    )
    base = json.loads(parent_path.read_text(encoding="utf-8"))
    base["formal_matrix"]["fault_scenarios"]["step"]["drift_delta"] = [
        9.0,
    ] * 5
    execution = subject._execution_config(base, config)
    assert execution["formal_matrix"]["fault_scenarios"] == config[
        "formal_matrix"
    ]["fault_scenario_parameters"]
    assert execution["formal_matrix"]["fault_scenarios"]["step"][
        "drift_delta"
    ] != [9.0] * 5


def test_application_evidence_builder_rejects_state_mutation() -> None:
    class Drift:
        def __init__(self, value: np.ndarray) -> None:
            self.value = np.asarray(value, dtype="<f8")

        def vector(self) -> np.ndarray:
            return self.value

    density = np.eye(3, dtype="<c16") / 3.0
    pre = SimpleNamespace(
        joint_density=density,
        cutoff=1,
        round_index=7,
        leakage_age=2,
        drift=Drift(np.arange(5, dtype="<f8")),
    )
    delta = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5], dtype="<f8")
    valid_input = SimpleNamespace(
        joint_density=density.copy(),
        cutoff=1,
        round_index=7,
        leakage_age=2,
        drift=Drift(pre.drift.vector() + delta),
    )
    identity = {
        "row_id": "producer-negative-r7",
        "scenario": "step",
        "round_index": 7,
    }
    evidence = subject._intervention_application_evidence(
        identity=identity,
        pre_state=pre,
        input_state=valid_input,
        delta=delta,
    )
    subject._validate_intervention_application_fields(
        {**identity, **evidence},
        delta,
    )
    wrong_drift = SimpleNamespace(
        **{
            **vars(valid_input),
            "drift": Drift(
                valid_input.drift.vector()
                + np.asarray([0.0, 0.0, 0.25, 0.0, 0.0], dtype="<f8")
            ),
        }
    )
    with pytest.raises(RuntimeError, match="drift application"):
        subject._intervention_application_evidence(
            identity=identity,
            pre_state=pre,
            input_state=wrong_drift,
            delta=delta,
        )
    wrong_non_drift = SimpleNamespace(
        **{
            **vars(valid_input),
            "joint_density": density
            + np.diag(np.asarray([0.01, -0.01, 0.0], dtype="<c16")),
        }
    )
    with pytest.raises(RuntimeError, match="mutated non-drift"):
        subject._intervention_application_evidence(
            identity=identity,
            pre_state=pre,
            input_state=wrong_non_drift,
            delta=delta,
        )


def test_real_backend_b_reset_capability_transaction_is_isolated(
    tmp_path: Path,
) -> None:
    config, binding = load_config(ROOT)
    plan = build_cell_plan(config)
    cell = next(
        item
        for item in plan
        if item.layer == "shared"
        and item.backend == "B"
        and item.cutoff == 36
        and item.initial_state == "vacuum_g"
        and item.action == "RESET"
    )
    receipt = execute_cell_to_store(
        root=ROOT,
        t04=config,
        config_sha256=str(binding["sha256"]),
        plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
        run_id="PYTEST-T04-CAPABILITY-RESET-V1",
        cell=cell,
        source_snapshot_sha256=_source_snapshot(),
        sample_count_override=1,
        seed_namespace="capability_preflight",
        artifact_paths_override=_isolated_paths(tmp_path),
    )
    assert receipt["diagnostics"] == {
        "conservation_failures": 0,
        "exception_rows": 0,
        "expected_rows": 1,
        "missing_rows": 0,
        "observed_rows": 1,
        "reset_rows": 1,
        "reset_sidecar_rows": 1,
    }
    objects = {item["role"]: item for item in receipt["objects"]}
    with (ROOT / objects["round_ledger_csv"]["path"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["reset_hidden_success"] == ""
    assert row["reset_ack"] == "marginalized"
    assert row["primary_reset_estimand"].startswith("RAO_BLACKWELLIZED_")
    assert row["sampled_reset_nonvoting"] == "True"
    assert row["intervention_delta_sha256"] == sha256(
        np.zeros(5, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    assert row["intervention_applied"] == "False"
    assert (
        row["input_state_sha256"]
        == row["pre_intervention_state_sha256"]
    )
    pre_drift = np.asarray(
        [float(row[name]) for name in PRE_DRIFT_FIELDS],
        dtype="<f8",
    )
    input_drift = np.asarray(
        [float(row[name]) for name in INPUT_DRIFT_FIELDS],
        dtype="<f8",
    )
    assert np.array_equal(pre_drift, input_drift)
    assert (
        row["pre_intervention_non_drift_state_sha256"]
        == row["input_non_drift_state_sha256"]
    )
    assert (
        row["intervention_application_receipt_sha256"]
        == _independent_application_receipt(row)
    )
    subject._validate_intervention_application_fields(
        row,
        np.zeros(5, dtype="<f8"),
    )
    _assert_intervention_mutations_rejected(
        row,
        np.zeros(5, dtype="<f8"),
    )
    assert int(row["heldout_seed_address"]) != heldout_seed(
        config, cell, 0, 0
    )
    ack = np.load(
        ROOT / objects["rb_sampled_reset_ack_npy"]["path"],
        allow_pickle=False,
    )
    assert ack.shape == (1,)
    assert ack[0].decode("ascii") in {"success", "failure"}


def test_real_mapping_anchor_artifact_matches_native_c36_isometries(
    tmp_path: Path,
) -> None:
    config, binding = load_config(ROOT)
    cell = build_cell_plan(config)[
        int(config["formal_matrix"]["mapping_anchor_plan_indices"]["36"])
    ]
    assert (cell.plan_index, cell.cutoff, cell.backend) == (0, 36, "A")
    receipt = execute_cell_to_store(
        root=ROOT,
        t04=config,
        config_sha256=str(binding["sha256"]),
        plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
        run_id="PYTEST-T04-MAPPING-ANCHOR-C36-V1",
        cell=cell,
        source_snapshot_sha256=_source_snapshot(),
        sample_count_override=1,
        seed_namespace="capability_preflight",
        artifact_paths_override=_isolated_paths(tmp_path),
    )
    objects = _objects(receipt)
    roles = {
        "mapping_isometry_a_npy",
        "mapping_isometry_b_npy",
        "mapping_projector_a_npy",
        "mapping_projector_b_npy",
    }
    assert roles <= set(objects)
    isometry_a = np.load(
        ROOT / str(objects["mapping_isometry_a_npy"]["path"]),
        allow_pickle=False,
    )
    isometry_b = np.load(
        ROOT / str(objects["mapping_isometry_b_npy"]["path"]),
        allow_pickle=False,
    )
    projector_a = np.load(
        ROOT / str(objects["mapping_projector_a_npy"]["path"]),
        allow_pickle=False,
    )
    projector_b = np.load(
        ROOT / str(objects["mapping_projector_b_npy"]["path"]),
        allow_pickle=False,
    )
    assert isometry_a.shape == isometry_b.shape == (36, 2)
    assert projector_a.shape == projector_b.shape == (36, 36)
    assert np.allclose(
        isometry_a.conj().T @ isometry_a,
        np.eye(2),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.allclose(
        isometry_b.conj().T @ isometry_b,
        np.eye(2),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.allclose(
        projector_a,
        isometry_a @ isometry_a.conj().T,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.allclose(
        projector_b,
        isometry_b @ isometry_b.conj().T,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_real_backend_b_c36_compound_continuation_has_two_reset_ancestors(
    tmp_path: Path,
) -> None:
    config, binding = load_config(ROOT)
    cell = next(
        item
        for item in build_cell_plan(config)
        if item.layer == "fault"
        and item.backend == "B"
        and item.cutoff == 36
        and item.scenario == "compound"
    )
    receipt = execute_cell_to_store(
        root=ROOT,
        t04=config,
        config_sha256=str(binding["sha256"]),
        plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
        run_id="PYTEST-T04-B-C36-COMPOUND-CONTINUATION-V1",
        cell=cell,
        source_snapshot_sha256=_source_snapshot(),
        sample_count_override=1,
        seed_namespace="capability_preflight",
        artifact_paths_override=_isolated_paths(tmp_path),
    )
    assert receipt["diagnostics"] == {
        "conservation_failures": 0,
        "exception_rows": 0,
        "expected_rows": 12,
        "missing_rows": 0,
        "observed_rows": 12,
        "reset_rows": 2,
        "reset_sidecar_rows": 2,
    }
    objects = _objects(receipt)
    rows = _read_rows(objects["round_ledger_csv"])
    assert len(rows) == 12
    assert [row["action"] for row in rows] == [
        "X",
        "Z",
        "IDLE",
        "XZ",
        "RESET",
        "HOLD",
        "X",
        "Z",
        "IDLE",
        "XZ",
        "RESET",
        "HOLD",
    ]
    for index, row in enumerate(rows):
        assert row["exception_type"] == ""
        assert row["conservation_pass"] == "True"
        assert row["input_evaluator_sha256"]
        assert row["output_evaluator_sha256"]
        if index:
            assert (
                row["pre_intervention_state_sha256"]
                == rows[index - 1]["output_state_sha256"]
            )
            assert (
                row["input_evaluator_sha256"]
                == rows[index - 1]["output_evaluator_sha256"]
            )
    first = rows[4]["expected_reset_ancestor_receipt_sha256"]
    second = rows[10]["expected_reset_ancestor_receipt_sha256"]
    assert len(first) == len(second) == 64
    assert first != second
    assert all(
        row["expected_reset_ancestor_receipt_sha256"] == ""
        for row in rows[:4]
    )
    assert all(
        row["expected_reset_ancestor_receipt_sha256"] == first
        for row in rows[4:10]
    )
    assert all(
        row["expected_reset_ancestor_receipt_sha256"] == second
        for row in rows[10:]
    )
    valid = np.load(
        ROOT / str(objects["rb_valid_npy"]["path"]),
        allow_pickle=False,
    )
    reset_rows = np.load(
        ROOT / str(objects["rb_row_index_npy"]["path"]),
        allow_pickle=False,
    )
    receipts = np.load(
        ROOT / str(objects["rb_pre_reset_receipt_npy"]["path"]),
        allow_pickle=False,
    )
    assert valid.tolist() == [True, True]
    assert reset_rows.tolist() == [4, 10]
    assert [value.decode("ascii") for value in receipts] == [first, second]


@pytest.mark.parametrize(
    "scenario",
    ["step", "telegraph", "burst", "compound"],
)
def test_real_fault_schedules_bind_intervention_and_continuation_witnesses(
    tmp_path: Path,
    scenario: str,
) -> None:
    config, binding = load_config(ROOT)
    cell = next(
        item
        for item in build_cell_plan(config)
        if item.layer == "fault"
        and item.backend == "B"
        and item.cutoff == 36
        and item.scenario == scenario
    )
    receipt = execute_cell_to_store(
        root=ROOT,
        t04=config,
        config_sha256=str(binding["sha256"]),
        plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
        run_id=f"PYTEST-T04-B-C36-{scenario.upper()}-WITNESS-V1",
        cell=cell,
        source_snapshot_sha256=_source_snapshot(),
        sample_count_override=1,
        seed_namespace="capability_preflight",
        artifact_paths_override=_isolated_paths(tmp_path),
    )
    expected_reset_rows = 2 if scenario == "compound" else 0
    assert receipt["diagnostics"] == {
        "conservation_failures": 0,
        "exception_rows": 0,
        "expected_rows": 12,
        "missing_rows": 0,
        "observed_rows": 12,
        "reset_rows": expected_reset_rows,
        "reset_sidecar_rows": expected_reset_rows,
    }
    rows = _read_rows(_objects(receipt)["round_ledger_csv"])
    specification = config["formal_matrix"]["fault_scenario_parameters"][
        scenario
    ]
    assert isinstance(specification, dict)
    assert len(rows) == int(specification["horizon"]) == 12
    for round_index, row in enumerate(rows):
        expected_delta = _expected_fault_delta(
            scenario,
            specification,
            round_index,
        )
        expected_applied = bool(np.any(expected_delta != 0.0))
        assert row["intervention_delta_sha256"] == sha256(
            np.asarray(expected_delta, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        assert row["intervention_applied"] == str(expected_applied)
        pre_drift = np.asarray(
            [float(row[name]) for name in PRE_DRIFT_FIELDS],
            dtype="<f8",
        )
        input_drift = np.asarray(
            [float(row[name]) for name in INPUT_DRIFT_FIELDS],
            dtype="<f8",
        )
        assert np.array_equal(input_drift, pre_drift + expected_delta)
        assert (
            row["pre_intervention_non_drift_state_sha256"]
            == row["input_non_drift_state_sha256"]
        )
        assert (
            row["intervention_application_receipt_sha256"]
            == _independent_application_receipt(row)
        )
        subject._validate_intervention_application_fields(
            row,
            expected_delta,
        )
        if round_index:
            assert (
                row["pre_intervention_state_sha256"]
                == rows[round_index - 1]["output_state_sha256"]
            )
        if expected_applied:
            assert (
                row["input_state_sha256"]
                != row["pre_intervention_state_sha256"]
            )
        else:
            assert (
                row["input_state_sha256"]
                == row["pre_intervention_state_sha256"]
            )

    applied_round = next(
        round_index
        for round_index, row in enumerate(rows)
        if row["intervention_applied"] == "True"
    )
    _assert_intervention_mutations_rejected(
        rows[applied_round],
        _expected_fault_delta(
            scenario,
            specification,
            applied_round,
        ),
    )

    ancestors = [
        row["expected_reset_ancestor_receipt_sha256"] for row in rows
    ]
    if scenario == "compound":
        first, second = ancestors[4], ancestors[10]
        assert len(first) == len(second) == 64
        assert first != second
        assert ancestors[:4] == [""] * 4
        assert ancestors[4:10] == [first] * 6
        assert ancestors[10:] == [second] * 2
    else:
        assert ancestors == [""] * 12


def test_mapping_anchor_nan_mutation_fails_and_releases_windows_memmaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, binding = load_config(ROOT)
    cell = build_cell_plan(config)[0]
    original = subject._stage_mapping_arrays

    def corrupted(*args: object, **kwargs: object) -> dict[str, Path]:
        paths = original(*args, **kwargs)
        target = paths["mapping_isometry_a_npy"]
        value = np.load(target, allow_pickle=False, mmap_mode="r+")
        value[0, 0] += 0.125
        value.flush()
        value._mmap.close()
        return paths

    monkeypatch.setattr(subject, "_stage_mapping_arrays", corrupted)
    isolated = _isolated_paths(tmp_path)
    with pytest.raises(RuntimeError, match="mapping anchor semantic"):
        execute_cell_to_store(
            root=ROOT,
            t04=config,
            config_sha256=str(binding["sha256"]),
            plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
            run_id="PYTEST-T04-MAPPING-MUTATION-V1",
            cell=cell,
            source_snapshot_sha256=_source_snapshot(),
            sample_count_override=1,
            seed_namespace="capability_preflight",
            artifact_paths_override=isolated,
        )
    staging = ROOT / isolated["staging_directory"]
    for path in staging.glob("*"):
        path.unlink()
    assert list(staging.glob("*")) == []


def test_rb_validation_exception_releases_open_windows_memmaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, binding = load_config(ROOT)
    cell = next(
        item
        for item in build_cell_plan(config)
        if item.layer == "shared"
        and item.backend == "B"
        and item.cutoff == 36
        and item.initial_state == "vacuum_g"
        and item.action == "RESET"
    )

    def rejected(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected RB validation failure")

    monkeypatch.setattr(subject, "_validate_rb_mixture", rejected)
    isolated = _isolated_paths(tmp_path)
    with pytest.raises(RuntimeError, match="injected RB"):
        execute_cell_to_store(
            root=ROOT,
            t04=config,
            config_sha256=str(binding["sha256"]),
            plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
            run_id="PYTEST-T04-RB-CLEANUP-MUTATION-V1",
            cell=cell,
            source_snapshot_sha256=_source_snapshot(),
            sample_count_override=1,
            seed_namespace="capability_preflight",
            artifact_paths_override=isolated,
        )
    staging = ROOT / isolated["staging_directory"]
    for path in staging.glob("*"):
        path.unlink()
    assert list(staging.glob("*")) == []


def test_fault_evaluator_carry_mutation_becomes_explicit_incomplete_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, binding = load_config(ROOT)
    cell = next(
        item
        for item in build_cell_plan(config)
        if item.layer == "fault"
        and item.backend == "B"
        and item.cutoff == 36
        and item.scenario == "compound"
    )
    monkeypatch.setattr(subject, "_next_evaluator", lambda result, backend: None)
    receipt = execute_cell_to_store(
        root=ROOT,
        t04=config,
        config_sha256=str(binding["sha256"]),
        plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
        run_id="PYTEST-T04-EVALUATOR-CARRY-MUTATION-V1",
        cell=cell,
        source_snapshot_sha256=_source_snapshot(),
        sample_count_override=1,
        seed_namespace="capability_preflight",
        artifact_paths_override=_isolated_paths(tmp_path),
    )
    diagnostics = receipt["diagnostics"]
    assert diagnostics["observed_rows"] == diagnostics["expected_rows"] == 12
    assert diagnostics["exception_rows"] == 12
    assert diagnostics["conservation_failures"] == 12
    assert diagnostics["reset_rows"] == diagnostics["reset_sidecar_rows"] == 2
    objects = _objects(receipt)
    rows = _read_rows(objects["round_ledger_csv"])
    assert all(
        row["exception_type"] == "RuntimeError"
        for row in rows
    )
    assert all(
        row["expected_reset_ancestor_receipt_sha256"] == ""
        for row in rows
    )
    valid = np.load(
        ROOT / str(objects["rb_valid_npy"]["path"]),
        allow_pickle=False,
    )
    assert valid.tolist() == [False, False]


def test_preflight_cannot_reuse_or_nest_formal_roots(tmp_path: Path) -> None:
    config, binding = load_config(ROOT)
    cell = build_cell_plan(config)[0]
    formal = config["artifact_paths"]
    bad = {
        "object_store": str(formal["object_store"]),
        "staging_directory": _relative(tmp_path / "staging"),
        "receipt_directory": _relative(tmp_path / "receipts"),
    }
    with pytest.raises(ValueError, match="overlaps formal"):
        execute_cell_to_store(
            root=ROOT,
            t04=config,
            config_sha256=str(binding["sha256"]),
            plan_sha256=str(plan_payload(config)["canonical_plan_sha256"]),
            run_id="PYTEST-T04-BAD-ISOLATION",
            cell=cell,
            source_snapshot_sha256=_source_snapshot(),
            sample_count_override=1,
            seed_namespace="capability_preflight",
            artifact_paths_override=bad,
        )


def test_formal_override_and_zero_source_are_rejected(tmp_path: Path) -> None:
    config, binding = load_config(ROOT)
    cell = build_cell_plan(config)[0]
    common = {
        "root": ROOT,
        "t04": config,
        "config_sha256": str(binding["sha256"]),
        "plan_sha256": str(plan_payload(config)["canonical_plan_sha256"]),
        "run_id": "PYTEST-T04-REJECT",
        "cell": cell,
        "artifact_paths_override": _isolated_paths(tmp_path),
    }
    with pytest.raises(ValueError, match="nonzero"):
        execute_cell_to_store(
            **common,
            source_snapshot_sha256="0" * 64,
            sample_count_override=1,
            seed_namespace="capability_preflight",
        )
    with pytest.raises(RuntimeError, match="preformal seal"):
        execute_cell_to_store(
            **common,
            source_snapshot_sha256=_source_snapshot(),
            sample_count_override=None,
            seed_namespace="formal",
        )


def test_failed_reset_sidecar_slot_is_explicit_and_nonvoting(
    tmp_path: Path,
) -> None:
    paths: dict[str, np.memmap] = {}
    arrays: dict[str, np.memmap] = {}
    specifications = {
        "valid": ((1,), "?"),
        "row_index": ((1,), "<i8"),
        "success_probability": ((1,), "<f8"),
        "success_present": ((1,), "?"),
        "failure_present": ((1,), "?"),
        "expected_density": ((1, 3, 3), "<c8"),
        "conditional_success_density": ((1, 3, 3), "<c8"),
        "conditional_failure_density": ((1, 3, 3), "<c8"),
        "sampled_stress_density": ((1, 3, 3), "<c8"),
        "sampled_hidden_outcome": ((1,), "u1"),
        "sampled_reset_ack": ((1,), "S16"),
        "branch_trace_distance": ((1,), "<f8"),
        "sampled_match_trace_distance": ((1,), "<f8"),
        "pre_reset_receipt": ((1,), "S64"),
    }
    for name, (shape, dtype) in specifications.items():
        path = tmp_path / f"{name}.npy"
        paths[name] = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=shape
        )
        arrays[name] = paths[name]
    _record_rb_failure(arrays, 0, row_index=17)
    assert not bool(arrays["valid"][0])
    assert int(arrays["row_index"][0]) == 17
    assert np.isnan(arrays["success_probability"][0])
    assert int(arrays["sampled_hidden_outcome"][0]) == 255
    assert np.isnan(arrays["expected_density"][0]).all()
