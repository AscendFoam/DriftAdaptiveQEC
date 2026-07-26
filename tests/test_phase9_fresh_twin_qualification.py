from __future__ import annotations

from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_fresh_twin_qualification as subject


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    value, _ = subject.load_config(ROOT)
    return value


@pytest.fixture(scope="module")
def actions():
    return subject._action_words()


@pytest.fixture(scope="module")
def c8_simulators(config):
    return subject.build_simulators(config, 8)


@pytest.fixture(scope="module")
def real_evidence(config, actions, c8_simulators):
    evidence = {}
    for backend in ("A", "B"):
        cell = subject.CellSpec(
            chunk_id=f"actual_{backend}",
            layer="shared",
            cell_base="shared|vacuum_g|IDLE",
            cutoff=8,
            backend=backend,
            sample_count=1,
            convergence_role="same_cutoff_ab+diagnostic_8_to_12",
            action="IDLE",
            initial_state="vacuum_g",
        )
        evidence[backend] = subject.execute_cell(
            config, cell, c8_simulators[backend], actions
        )
    return evidence


def _mutated(config, path, value):
    result = deepcopy(config)
    cursor = result
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return result


def _minimal_row(row_id: str, chunk_id: str, index: int) -> dict[str, object]:
    row = {field: "" for field in subject.LEDGER_FIELDS}
    row.update(
        {
            "row_id": row_id,
            "row_schema": subject.ROW_SCHEMA,
            "layer": "shared",
            "cell_base": "shared|vacuum_g|IDLE",
            "cell_id": "ab/c8/shared/vacuum_g/IDLE",
            "backend": "A",
            "backend_id": "test",
            "cutoff": 8,
            "convergence_role": "same_cutoff_ab",
            "seed": 1070000,
            "seed_position": index,
            "round_index": 0,
            "terminal_round": True,
            "action": "IDLE",
            "rng_namespace": "test",
            "archive_chunk": chunk_id,
            "archive_row_index": index,
            "density_index": index,
            "raw_iq_index": index,
            "heldout_iq_index": index,
            "reset_requested": False,
            "reset_hidden_success": False,
            "leakage_resident": False,
            "conservation_pass": True,
            "exception_type": "",
            "exception_message": "",
        }
    )
    return row


def _bind_synthetic_archive_fields(
    row: dict[str, object],
    *,
    heldout: np.ndarray,
    density: np.ndarray,
) -> None:
    row["heldout_window_sha256"] = sha256(
        np.asarray(heldout, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    u = 2.0**-24
    certified = (
        u / (1.0 - u) * np.linalg.norm(density.astype(np.complex128), ord="fro")
        + np.sqrt(2.0 * density.size) * 2.0**-150
    )
    row["density_quantization_frobenius_error"] = 0.0
    row["density_quantization_certified_frobenius_bound"] = certified
    row["density_quantization_trace_distance_bound"] = (
        0.5 * np.sqrt(density.shape[0]) * certified
    )


def _signed(document: dict[str, object]) -> dict[str, object]:
    result = dict(document)
    result["analysis_sha256"] = sha256(subject._canonical(result)).hexdigest()
    return result


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sealed_fixture_root(tmp_path: Path, config) -> tuple[dict, dict]:
    local = deepcopy(config)
    config_path = tmp_path / subject.CONFIG_PATH
    _write_json(config_path, local)
    runner_path = (
        tmp_path / "cnn_fpga/benchmark/phase9_fresh_twin_qualification.py"
    )
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_bytes(
        (
            ROOT / "cnn_fpga/benchmark/phase9_fresh_twin_qualification.py"
        ).read_bytes()
    )
    for relative in local["runtime_dependencies"]["paths"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    lineage = _signed(
        {
            "task_id": subject.TASK_ID,
            "schema_version": "1.0",
            "verdict": "PASS_HISTORICAL_NO_GO_LINEAGE_BOUND",
        }
    )
    design = _signed(
        {
            "task_id": subject.TASK_ID,
            "schema_version": "1.0",
            "verdict": "PASS_FRESH_TWIN_DESIGN_POWER",
            "blueprint": {
                "gate_count": 1589,
                "sha256": local["gate_blueprint"][
                    "source_design_blueprint_sha256"
                ],
            },
        }
    )
    _write_json(
        tmp_path
        / local["historical_policy"]["historical_lineage_receipt"]["path"],
        lineage,
    )
    _write_json(tmp_path / local["design_power"]["path"], design)
    audit_path = tmp_path / local["preformal_seal"]["audit_path"]
    _write_json(
        audit_path,
        _signed(
            {
                "task_id": subject.TASK_ID,
                "schema_version": "fixture-audit",
                "status": "PASS",
            }
        ),
    )
    config_binding = subject._binding(config_path, tmp_path)
    paths = (
        subject.CONFIG_PATH,
        "cnn_fpga/benchmark/phase9_fresh_twin_qualification.py",
        local["historical_policy"]["historical_lineage_receipt"]["path"],
        local["design_power"]["path"],
        local["preformal_seal"]["audit_path"],
        *local["runtime_dependencies"]["paths"],
    )
    live = {
        f"binding_{index}": subject._binding_for_relative(tmp_path, path)
        for index, path in enumerate(paths)
    }
    seal = _signed(
        {
            "task_id": subject.TASK_ID,
            "schema_version": subject.PREFORMAL_SEAL_SCHEMA,
            "status": local["preformal_seal"]["required_status"],
            "formal_result_accessed": False,
            "historical_formal_cell_data_accessed": False,
            "all_gates_passed": True,
            "all_mutations_detected": True,
            "scientific_verdict": None,
            "live_bindings": live,
        }
    )
    _write_json(tmp_path / local["preformal_seal"]["path"], seal)
    return local, config_binding


def test_frozen_public_schema_constants_are_distinct():
    values = {
        subject.CONFIG_SCHEMA,
        subject.RUNNER_ID,
        subject.ROW_SCHEMA,
        subject.RAW_ARCHIVE_SCHEMA,
        subject.ATTEMPT_SCHEMA,
        subject.MANIFEST_SCHEMA,
        subject.HEARTBEAT_SCHEMA,
    }
    assert len(values) == 7


def test_config_loads_and_has_no_scientific_cli_override(config):
    assert config["task_id"] == subject.TASK_ID
    assert config["runner_policy"]["formal_cli_overrides"] == []
    assert config["runner_policy"]["runner_emits_scientific_verdict"] is False
    assert config["claim_boundary"]["runner_qualified_claim"] is None
    assert len(config["claim_boundary"]["current_claim_state"]) == 15
    assert all(
        value is None
        for value in config["claim_boundary"]["current_claim_state"].values()
    )


def test_plan_has_exact_powered_accounting(config):
    plan = subject.build_cell_plan(config)
    assert len(plan) == 592
    assert sum(cell.expected_rows for cell in plan) == 528_384


def test_materialized_gate_blueprint_is_complete_and_hash_bound(config):
    blueprint = config["gate_blueprint"]
    assert blueprint["row_count"] == len(blueprint["rows"]) == 1589
    payload = json.dumps(
        blueprint["rows"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert sha256(payload).hexdigest() == blueprint["canonical_blueprint_sha256"]
    assert blueprint["source_design_blueprint_sha256"] == (
        "bf586a2f5ba6096ad4446f1c18b30eeff4d0cd13c9ef523ed482dd293d76e24b"
    )
    assert {
        row["direction"]
        for row in blueprint["rows"]
        if row["metric"] == "principal_singular"
    } == {"lower"}
    assert all(
        row["direction"] == "upper"
        for row in blueprint["rows"]
        if row["metric"] != "principal_singular"
    )


def test_plan_layer_cell_counts(config):
    plan = subject.build_cell_plan(config)
    counts = {
        layer: sum(cell.layer == layer for cell in plan)
        for layer in ("shared", "probe", "logical", "fault")
    }
    assert counts == {"shared": 240, "probe": 32, "logical": 288, "fault": 32}


def test_c20_is_tail_submatrix_only(config):
    c20 = [cell for cell in subject.build_cell_plan(config) if cell.cutoff == 20]
    assert len(c20) == 74
    assert all(
        cell.action in {"IDLE", "XZ", "RESET"}
        for cell in c20
        if cell.layer in {"shared", "logical"}
    )
    assert all("primary_16_to_20" in cell.convergence_role for cell in c20)


def test_c8_to_c12_is_diagnostic_while_later_pairs_are_primary(config):
    plan = subject.build_cell_plan(config)
    assert all(
        "diagnostic_8_to_12" in cell.convergence_role
        for cell in plan
        if cell.cutoff in {8, 12}
    )
    assert all(
        "primary_12_to_16" in cell.convergence_role
        for cell in plan
        if cell.cutoff in {12, 16}
    )


def test_all_representative_probes_are_present_for_both_backends(config):
    probe_cells = [
        cell for cell in subject.build_cell_plan(config) if cell.layer == "probe"
    ]
    assert len(probe_cells) == 32
    assert len({cell.probe_id for cell in probe_cells}) == 16
    assert {cell.backend for cell in probe_cells} == {"A", "B"}


def test_blueprint_cell_id_is_exact(config):
    cell = next(
        cell
        for cell in subject.build_cell_plan(config)
        if cell.layer == "shared"
        and cell.cutoff == 8
        and cell.initial_state == "vacuum_g"
        and cell.action == "IDLE"
        and cell.backend == "A"
    )
    identity = subject._identity(
        cell, seed=1070000, position=0, row_index=0, action="IDLE"
    )
    assert identity["cell_id"] == "ab/c8/shared/vacuum_g/IDLE"


def test_heldout_window_is_backend_common_and_address_stable(config):
    first = subject._heldout_window(
        config,
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        position=0,
        round_index=0,
    )
    second = subject._heldout_window(
        config,
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        position=0,
        round_index=0,
    )
    assert np.array_equal(first, second)


def test_heldout_address_changes_with_cell_or_round(config):
    base = subject._heldout_window(
        config,
        cell_base="fault|step",
        cutoff=8,
        position=0,
        round_index=0,
    )
    other = subject._heldout_window(
        config,
        cell_base="fault|step",
        cutoff=8,
        position=0,
        round_index=1,
    )
    assert not np.array_equal(base, other)


def test_heldout_record_is_paired_across_cutoffs(config):
    low = subject._heldout_window(
        config,
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        position=7,
        round_index=0,
    )
    high = subject._heldout_window(
        config,
        cell_base="shared|vacuum_g|IDLE",
        cutoff=20,
        position=7,
        round_index=0,
    )
    assert np.array_equal(low, high)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("design_power", "round_sample_count"), 767),
        (("formal_matrix", "round_sample_count"), 512),
        (("formal_matrix", "trajectory_sample_count"), 192),
        (("formal_matrix", "same_cutoff_ab"), [8, 12]),
        (("formal_matrix", "primary_cutoff_increments"), [[8, 12], [12, 16]]),
        (("formal_matrix", "no_postselection"), False),
        (("readout_semantics", "raw_log_evidence_primary"), True),
        (("resume_policy", "nan_to_zero"), True),
        (("runner_policy", "formal_cli_overrides"), ["seed"]),
        (("gate_blueprint", "row_count"), 1588),
    ],
)
def test_config_scientific_mutations_fail_closed(config, path, value):
    with pytest.raises(ValueError):
        subject.validate_config(_mutated(config, path, value))


def test_boolean_cannot_alias_seed_integer(config):
    mutant = _mutated(
        config, ("formal_splits", "round_backend_a", "start"), True
    )
    with pytest.raises(ValueError):
        subject.validate_config(mutant)


def test_overlapping_seed_intervals_fail_closed(config):
    mutant = _mutated(
        config, ("formal_splits", "round_backend_b", "start"), 1070500
    )
    with pytest.raises(ValueError, match="overlap"):
        subject.validate_config(mutant)


def test_artifact_path_escape_fails_closed(config):
    mutant = _mutated(
        config, ("artifact_paths", "raw_archive"), "../escape.zip"
    )
    with pytest.raises(ValueError, match="escapes"):
        subject.validate_config(mutant)


def test_preformal_seal_exact_live_bindings_are_accepted(tmp_path, config):
    local, binding = _sealed_fixture_root(tmp_path, config)
    seal, seal_binding = subject.verify_preformal_seal(
        tmp_path, local, binding
    )
    assert seal["all_gates_passed"] is True
    assert seal_binding["path"] == local["preformal_seal"]["path"]


def test_preformal_seal_status_mutation_fails_closed(tmp_path, config):
    local, binding = _sealed_fixture_root(tmp_path, config)
    path = tmp_path / local["preformal_seal"]["path"]
    seal = json.loads(path.read_text(encoding="utf-8"))
    seal["status"] = "NO_GO"
    unsigned = dict(seal)
    unsigned.pop("analysis_sha256")
    seal["analysis_sha256"] = sha256(subject._canonical(unsigned)).hexdigest()
    _write_json(path, seal)
    with pytest.raises(RuntimeError, match="policy mismatch"):
        subject.verify_preformal_seal(tmp_path, local, binding)


def test_preformal_live_binding_drift_fails_closed(tmp_path, config):
    local, binding = _sealed_fixture_root(tmp_path, config)
    audit = tmp_path / local["preformal_seal"]["audit_path"]
    audit.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drift"):
        subject.verify_preformal_seal(tmp_path, local, binding)


def test_fresh_sources_do_not_name_historical_cell_artifacts():
    source = (
        ROOT / "cnn_fpga/benchmark/phase9_fresh_twin_qualification.py"
    ).read_text(encoding="utf-8")
    config_source = (
        ROOT / "configs/phase9/t_risk_20260726_01_fresh_twin_qualification.json"
    ).read_text(encoding="utf-8")
    prohibited = (
        "t9_2_4_dual_backend_cell_" + "ledger.csv",
        "t9_2_4_dual_backend_qualification_" + "source_data.csv",
        "t9_2_4_dual_backend_state_" + "archive.npz",
    )
    assert all(name not in source + config_source for name in prohibited)


def test_real_a_and_b_rows_use_same_heldout_record(real_evidence):
    assert np.array_equal(
        real_evidence["A"].heldout_iq, real_evidence["B"].heldout_iq
    )
    assert (
        real_evidence["A"].rows[0]["heldout_window_sha256"]
        == real_evidence["B"].rows[0]["heldout_window_sha256"]
    )


@pytest.mark.parametrize("backend", ["A", "B"])
def test_real_backend_row_matches_independent_likelihood_reference(
    real_evidence, backend
):
    row = real_evidence[backend].rows[0]
    assert row["exception_type"] == ""
    assert row["reference_log_evidence_error"] <= 5.0e-10
    assert row["reference_posterior_l1_error"] <= 5.0e-10
    assert row["integrated_i_mean_error"] <= 2.0e-12
    assert row["integrated_q_mean_error"] <= 2.0e-12
    assert row["conservation_pass"] is True


@pytest.mark.parametrize("backend", ["A", "B"])
def test_real_shared_row_retains_complex64_density_with_certificate(
    real_evidence, backend
):
    evidence = real_evidence[backend]
    row = evidence.rows[0]
    density = evidence.densities[0]
    assert density.dtype == np.complex64
    u = 2.0**-24
    expected = (
        u / (1.0 - u) * np.linalg.norm(density.astype(np.complex128), ord="fro")
        + np.sqrt(2.0 * density.size) * 2.0**-150
    )
    assert row["density_quantization_certified_frobenius_bound"] == pytest.approx(
        expected, rel=1e-14
    )
    assert row["density_quantization_frobenius_error"] <= expected
    assert row["density_quantization_trace_distance_bound"] == pytest.approx(
        0.5 * np.sqrt(density.shape[0]) * expected, rel=1e-14
    )


def test_logical_rows_do_not_duplicate_joint_density(config, actions, c8_simulators):
    cell = subject.CellSpec(
        chunk_id="logical_actual",
        layer="logical",
        cell_base="logical|0|IDLE",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        action="IDLE",
        logical_label="0",
    )
    evidence = subject.execute_cell(config, cell, c8_simulators["A"], actions)
    assert evidence.rows[0]["density_index"] == -1
    assert evidence.densities == []
    assert evidence.rows[0]["logical_survival"] != ""
    assert evidence.rows[0]["logical_block_00_real"] != ""


def test_fault_retains_only_terminal_density(config, actions, c8_simulators):
    cell = subject.CellSpec(
        chunk_id="fault_actual",
        layer="fault",
        cell_base="fault|step",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        scenario="step",
        horizon=2,
    )
    evidence = subject.execute_cell(config, cell, c8_simulators["A"], actions)
    assert [row["density_index"] for row in evidence.rows] == [-1, 0]
    assert len(evidence.densities) == 1
    assert evidence.density_row_ids == [evidence.rows[-1]["row_id"]]
    assert {row["logical_label"] for row in evidence.rows} == {"0"}
    assert evidence.rows[-1]["logical_survival"] != ""
    assert evidence.rows[-1]["logical_block_00_real"] != ""


def test_fault_six_state_schedule_is_balanced_and_frozen(config):
    schedule = config["formal_matrix"]["fault_logical_label_schedule"]
    labels = [schedule[position % len(schedule)] for position in range(256)]
    counts = {label: labels.count(label) for label in schedule}
    assert set(counts.values()) == {42, 43}
    assert max(counts.values()) - min(counts.values()) == 1


def test_reset_success_is_rao_blackwellized(config, actions, c8_simulators):
    cell = subject.CellSpec(
        chunk_id="reset_actual",
        layer="shared",
        cell_base="shared|vacuum_e|RESET",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        action="RESET",
        initial_state="vacuum_e",
    )
    row = subject.execute_cell(
        config, cell, c8_simulators["A"], actions
    ).rows[0]
    expected = (
        row["pre_reset_g"]
        + config["common_physics"]["reset_success_e"] * row["pre_reset_e"]
        + config["common_physics"]["reset_success_f"] * row["pre_reset_f"]
    )
    assert row["reset_requested"] is True
    assert row["rao_blackwell_reset_success"] == pytest.approx(expected)


def test_exception_row_keeps_full_identity_and_heldout_without_fake_raw(
    config, actions
):
    class BrokenSimulator:
        def initialize_fock(self, *args, **kwargs):
            raise RuntimeError("injected failure")

    cell = subject.CellSpec(
        chunk_id="broken",
        layer="shared",
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        action="IDLE",
        initial_state="vacuum_g",
    )
    evidence = subject.execute_cell(config, cell, BrokenSimulator(), actions)
    row = evidence.rows[0]
    assert row["row_id"]
    assert row["cell_id"] == "ab/c8/shared/vacuum_g/IDLE"
    assert row["heldout_iq_index"] == 0
    assert len(row["heldout_window_sha256"]) == 64
    assert row["raw_iq_index"] == -1
    assert np.isnan(evidence.raw_iq[0]).all()
    assert np.isfinite(evidence.heldout_iq[0]).all()
    assert row["exception_type"] == "RuntimeError"


def test_attempt_ledger_is_append_only_hash_chain(tmp_path):
    path = tmp_path / "attempts.jsonl"
    first = subject._append_event(path, {"event_kind": "RUN_STARTED"})
    before = path.read_bytes()
    second = subject._append_event(path, {"event_kind": "RUN_ERROR"})
    after = path.read_bytes()
    events, payload = subject._parse_attempt_ledger(path)
    assert after.startswith(before)
    assert payload == after
    assert events == [first, second]
    assert second["previous_event_sha256"] == first["event_sha256"]


def test_torn_attempt_ledger_fails_closed(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b'{"event_kind":"RUN_STARTED"}')
    with pytest.raises(RuntimeError, match="torn"):
        subject._parse_attempt_ledger(path)


def test_mutated_attempt_event_hash_fails_closed(tmp_path):
    path = tmp_path / "attempts.jsonl"
    subject._append_event(path, {"event_kind": "RUN_STARTED"})
    payload = path.read_bytes().replace(b"RUN_STARTED", b"RUN_MUTATED")
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="self-hash"):
        subject._parse_attempt_ledger(path)


def test_exact_resume_accepts_only_hash_bound_semantic_chunk(tmp_path, config):
    local = deepcopy(config)
    local["artifact_paths"]["chunk_directory"] = "chunks"
    local["artifact_paths"]["attempt_ledger"] = "attempts.jsonl"
    config_file = tmp_path / "config.json"
    seal_file = tmp_path / "seal.json"
    config_file.write_text("{}\n", encoding="utf-8")
    seal_file.write_text("{}\n", encoding="utf-8")
    config_binding = subject._binding(config_file, tmp_path)
    seal_binding = subject._binding(seal_file, tmp_path)
    cell = subject.CellSpec(
        chunk_id="resume_chunk",
        layer="shared",
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        action="IDLE",
        initial_state="vacuum_g",
    )
    row = _minimal_row("resume-r0", cell.chunk_id, 0)
    density = np.eye(24, dtype=np.complex64) / 24
    heldout = np.ones((8, 2), dtype=np.float64)
    _bind_synthetic_archive_fields(row, heldout=heldout, density=density)
    receipt = subject.write_chunk(
        tmp_path,
        local,
        cell,
        subject.ChunkEvidence(
            rows=[row],
            densities=[density],
            density_row_ids=["resume-r0"],
            raw_iq=np.zeros((1, 8, 2)),
            heldout_iq=heldout[None, :, :],
        ),
    )
    run_id = subject._run_id(config_binding, seal_binding)
    common = {
        "task_id": subject.TASK_ID,
        "runner_id": subject.RUNNER_ID,
        "run_id": run_id,
        "config_sha256": config_binding["sha256"],
        "seal_sha256": seal_binding["sha256"],
    }
    attempt = tmp_path / "attempts.jsonl"
    subject._append_event(
        attempt, {**common, "event_kind": "RUN_STARTED"}
    )
    subject._append_event(
        attempt,
        {**common, "event_kind": "CHUNK_COMMITTED", "chunk": receipt},
    )
    observed_id, committed, _ = subject._resume_state(
        tmp_path, local, config_binding, seal_binding, [cell]
    )
    assert observed_id == run_id
    assert committed[cell.chunk_id] == receipt
    npz = tmp_path / receipt["npz"]["path"]
    npz.write_bytes(npz.read_bytes() + b"mutated")
    with pytest.raises(RuntimeError, match="committed chunk drift"):
        subject._resume_state(
            tmp_path, local, config_binding, seal_binding, [cell]
        )


def test_chunk_npz_is_pickle_free_and_row_aligned(tmp_path, config):
    local = deepcopy(config)
    local["artifact_paths"]["chunk_directory"] = "chunks"
    cell = subject.CellSpec(
        chunk_id="synthetic",
        layer="shared",
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        action="IDLE",
        initial_state="vacuum_g",
    )
    row = _minimal_row("r0", cell.chunk_id, 0)
    density = np.eye(24, dtype=np.complex64) / 24
    heldout = np.ones((8, 2), dtype=np.float64)
    _bind_synthetic_archive_fields(row, heldout=heldout, density=density)
    evidence = subject.ChunkEvidence(
        rows=[row],
        densities=[density],
        density_row_ids=["r0"],
        raw_iq=np.zeros((1, 8, 2), dtype=np.float64),
        heldout_iq=heldout[None, :, :],
    )
    receipt = subject.write_chunk(tmp_path, local, cell, evidence)
    with np.load(
        tmp_path / receipt["npz"]["path"], allow_pickle=False
    ) as archive:
        assert archive["schema"][0] == subject.RAW_ARCHIVE_SCHEMA
        assert archive["row_ids"].tolist() == ["r0"]
        assert archive["density_row_ids"].tolist() == ["r0"]
        assert archive["raw_iq"].shape == (1, 8, 2)
        assert archive["densities"].dtype == np.complex64


def test_quantization_certificate_mutation_is_detected(tmp_path, config):
    local = deepcopy(config)
    local["artifact_paths"]["chunk_directory"] = "chunks"
    cell = subject.CellSpec(
        chunk_id="quant_mutant",
        layer="shared",
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        action="IDLE",
        initial_state="vacuum_g",
    )
    row = _minimal_row("r0", cell.chunk_id, 0)
    density = np.eye(24, dtype=np.complex64) / 24
    heldout = np.ones((8, 2), dtype=np.float64)
    _bind_synthetic_archive_fields(row, heldout=heldout, density=density)
    evidence = subject.ChunkEvidence(
        rows=[row],
        densities=[density],
        density_row_ids=["r0"],
        raw_iq=np.zeros((1, 8, 2)),
        heldout_iq=heldout[None, :, :],
    )
    receipt = subject.write_chunk(tmp_path, local, cell, evidence)
    csv_path = tmp_path / receipt["csv"]["path"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["density_quantization_certified_frobenius_bound"] = "0"
    subject._atomic_csv(csv_path, rows)
    with pytest.raises(RuntimeError, match="quantization certificate"):
        subject._validate_chunk_files(tmp_path, receipt, cell)


def test_final_raw_zip_is_self_contained_and_hash_bound(tmp_path, config):
    local = deepcopy(config)
    local["artifact_paths"]["chunk_directory"] = "chunks"
    local["artifact_paths"]["raw_archive"] = "raw.zip"
    cell = subject.CellSpec(
        chunk_id="synthetic",
        layer="shared",
        cell_base="shared|vacuum_g|IDLE",
        cutoff=8,
        backend="A",
        sample_count=1,
        convergence_role="same_cutoff_ab",
        action="IDLE",
        initial_state="vacuum_g",
    )
    row = _minimal_row("r0", cell.chunk_id, 0)
    density = np.eye(24, dtype=np.complex64) / 24
    heldout = np.ones((8, 2), dtype=np.float64)
    _bind_synthetic_archive_fields(row, heldout=heldout, density=density)
    evidence = subject.ChunkEvidence(
        rows=[row],
        densities=[density],
        density_row_ids=["r0"],
        raw_iq=np.zeros((1, 8, 2)),
        heldout_iq=heldout[None, :, :],
    )
    receipt = subject.write_chunk(tmp_path, local, cell, evidence)
    mapping = tmp_path / "chunks/mapping.npz"
    subject._atomic_npz(
        mapping,
        schema=np.asarray([subject.RAW_ARCHIVE_SCHEMA]),
        mapping=np.eye(2, dtype=np.complex128),
    )
    binding = subject._build_raw_archive(
        tmp_path,
        local,
        [cell],
        {cell.chunk_id: receipt},
        subject._binding(mapping, tmp_path),
    )
    with zipfile.ZipFile(tmp_path / binding["path"], "r") as archive:
        manifest = json.loads(archive.read("archive_manifest.json"))
        member = archive.read(manifest["entries"][0]["member"])
        assert sha256(member).hexdigest() == receipt["npz"]["sha256"]
        with np.load(io.BytesIO(member), allow_pickle=False) as raw:
            assert raw["row_ids"].tolist() == ["r0"]
        assert "mapping/mapping_arrays.npz" in archive.namelist()


def test_merged_ledger_is_complete_and_deterministic(tmp_path, config):
    local = deepcopy(config)
    local["artifact_paths"]["chunk_directory"] = "chunks"
    local["artifact_paths"]["cell_ledger"] = "ledger.csv"
    cells = []
    receipts = {}
    for index in range(2):
        cell = subject.CellSpec(
            chunk_id=f"chunk{index}",
            layer="shared",
            cell_base=f"shared|vacuum_g|{'IDLE' if index == 0 else 'X'}",
            cutoff=8,
            backend="A",
            sample_count=1,
            convergence_role="same_cutoff_ab",
            action="IDLE" if index == 0 else "X",
            initial_state="vacuum_g",
        )
        row = _minimal_row(f"r{index}", cell.chunk_id, 0)
        density = np.eye(24, dtype=np.complex64) / 24
        heldout = np.zeros((8, 2), dtype=np.float64)
        _bind_synthetic_archive_fields(row, heldout=heldout, density=density)
        evidence = subject.ChunkEvidence(
            rows=[row],
            densities=[density],
            density_row_ids=[f"r{index}"],
            raw_iq=np.zeros((1, 8, 2)),
            heldout_iq=heldout[None, :, :],
        )
        receipts[cell.chunk_id] = subject.write_chunk(
            tmp_path, local, cell, evidence
        )
        cells.append(cell)
    subject._merge_ledger(tmp_path, local, cells, receipts)
    with (tmp_path / "ledger.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["row_id"] for row in rows] == ["r0", "r1"]


def test_main_rejects_any_seed_override():
    with pytest.raises(SystemExit):
        subject.main(["--seed", "1"])
