from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_cutoff36_44_repair as subject
from cnn_fpga.benchmark import (
    phase9_cutoff36_44_repair_diagnostic as diagnostic,
)
from cnn_fpga.benchmark import (
    phase9_cutoff36_44_repair_verify as verifier,
)
from cnn_fpga.benchmark import phase9_fresh_twin_qualification as fresh
import physics.phase9_reset_rao_blackwell as rb


ROOT = Path(__file__).resolve().parents[1]


def test_config_is_self_hashed_and_freezes_terminal_bounded_repair() -> None:
    config, _base = subject.load_config(ROOT)
    assert config["analysis_sha256"] == subject._self_hash(config)
    assert config["cutoffs"] == [36, 40, 44]
    assert config["required_consecutive_increments"] == [[36, 40], [40, 44]]
    assert config["absolute_tail_cutoff"] == 44
    assert config["automatic_cutoff_extension_beyond_44"] is False
    assert config["terminal_if_cutoff44_fails"] is True
    assert config["repair_trigger"]["old_raw_rows_vote"] is False
    assert config["repair_trigger"]["old_passing_gates_vote"] is False
    assert config["future_powered_formal"]["released"] is False
    assert all(
        config["claim_boundary"][field] is None
        for field in (
            "twin_qualification",
            "ler",
            "lifetime",
            "physical_break_even",
            "official_puviani_exact",
            "puviani_nmf_surpass",
            "external_sota",
            "hardware_measured",
        )
    )


def test_config_mutations_fail_closed() -> None:
    config, _base = subject.load_config(ROOT)
    for path, value in (
        (("cutoffs",), [36, 40, 44, 48]),
        (("automatic_cutoff_extension_beyond_44",), True),
        (("repair_trigger", "old_passing_gates_vote"), True),
        (("future_powered_formal", "released"), True),
    ):
        mutated = deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        mutated["analysis_sha256"] = subject._self_hash(mutated)
        with pytest.raises(RuntimeError):
            subject._validate_config(ROOT, mutated)


def test_cell_plan_has_exact_fresh_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, base = subject.load_config(ROOT)
    monkeypatch.setattr(subject, "runner", fresh)
    execution = subject.materialize_execution(config, base)
    cells = subject.build_cells(config, execution)
    assert len(cells) == 30
    assert sum(cell.expected_rows for cell in cells) == 21_168
    assert len({cell.chunk_id for cell in cells}) == 30
    assert {
        (cell.cutoff, cell.backend, cell.layer)
        for cell in cells
        if cell.layer == "shared"
    } == {
        (cutoff, backend, "shared") for cutoff in (36, 40, 44) for backend in ("A", "B")
    }


def test_verified_loader_activates_cap44_in_isolated_process() -> None:
    script = (
        "from cnn_fpga.benchmark import phase9_cutoff36_44_repair as s;"
        "r=s._root();c,b=s.load_config(r);s._activate_verified_modules(r,c);"
        "e=s.materialize_execution(c,b);x=s.build_cells(c,e);"
        "print(len(x),sum(v.expected_rows for v in x),"
        "s._ADAPTER_RECEIPT['extended_max_supported_cutoff'])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.strip() == "30 21168 44"


@pytest.mark.parametrize("backend", ["A", "B"])
def test_shared_writer_uses_expected_density_and_archives_nonvoting_branches(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, base = subject.load_config(ROOT)
    monkeypatch.setattr(subject, "runner", fresh)
    monkeypatch.setattr(subject, "rb_reset", rb)
    execution = subject.materialize_execution(config, base)
    cell = fresh.CellSpec(
        chunk_id=f"unit_rb_{backend}",
        layer="shared",
        cell_base="shared|vacuum_f|RESET",
        cutoff=8,
        backend=backend,
        sample_count=2,
        convergence_role="unit_test",
        action="RESET",
        initial_state="vacuum_f",
    )
    simulator = fresh.build_simulators(execution, 8)[backend]
    evidence, sidecar = subject._execute_shared_rb(
        config,
        execution,
        cell,
        simulator,
    )
    assert len(evidence.rows) == 2
    assert len(evidence.densities) == 2
    assert sidecar["conditional_success_densities"].shape == (2, 24, 24)
    assert sidecar["conditional_failure_densities"].shape == (2, 24, 24)
    assert sidecar["sampled_stress_densities"].shape == (2, 24, 24)
    probabilities = sidecar["success_probability"]
    mixture = (
        probabilities[:, None, None] * sidecar["conditional_success_densities"]
        + (1.0 - probabilities[:, None, None])
        * sidecar["conditional_failure_densities"]
    )
    assert np.allclose(
        np.stack(evidence.densities),
        mixture,
        rtol=0.0,
        atol=3.0e-6,
    )
    for row, density in zip(evidence.rows, evidence.densities, strict=True):
        assert row["conservation_pass"] is True
        assert np.allclose(
            [row["level_g"], row["level_e"], row["level_f"]],
            simulator.level_probabilities(density),
            atol=3.0e-6,
        )
        assert row["logical_survival"] == ""
        assert row["reset_hidden_success"] in {True, False}


def test_help_is_zero_write() -> None:
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (ROOT / "runs/t_risk_20260728_03_cutoff36_44_repair_fresh1").glob(
            "**/*"
        )
        if path.is_file()
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cnn_fpga.benchmark.phase9_cutoff36_44_repair",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (ROOT / "runs/t_risk_20260728_03_cutoff36_44_repair_fresh1").glob(
            "**/*"
        )
        if path.is_file()
    }
    assert "--preflight-only" in completed.stdout
    assert before == after


def test_diagnostic_enumerates_only_two_fresh_increments_and_cutoff44(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _base = subject.load_config(ROOT)
    monkeypatch.setattr(diagnostic.legacy, "_density_point", lambda *args: 0.0)
    monkeypatch.setattr(diagnostic.legacy, "_trace_distance", lambda *args: 0.0)
    monkeypatch.setattr(
        diagnostic.legacy,
        "_tail_features",
        lambda stack, cutoff: {
            "top1_fock_mass": np.zeros(len(stack)),
            "top2_fock_mass": np.zeros(len(stack)),
            "top4_fock_mass": np.zeros(len(stack)),
            "normalized_mean_photon": np.zeros(len(stack)),
            "commutator_defect": np.zeros(len(stack)),
        },
    )
    monkeypatch.setattr(
        diagnostic.legacy,
        "_tail_quantization_bound",
        lambda *args: 0.0,
    )
    monkeypatch.setattr(verifier, "_density_point", lambda *args: 0.0)
    monkeypatch.setattr(verifier, "_trace_distance", lambda *args: 0.0)
    monkeypatch.setattr(
        verifier,
        "_tail_features",
        lambda stack, cutoff: {
            "top1_fock_mass": np.zeros(len(stack)),
            "top2_fock_mass": np.zeros(len(stack)),
            "top4_fock_mass": np.zeros(len(stack)),
            "normalized_mean_photon": np.zeros(len(stack)),
            "commutator_defect": np.zeros(len(stack)),
        },
    )
    monkeypatch.setattr(verifier, "_tail_quantization", lambda *args: 0.0)
    rows: list[dict[str, object]] = []
    densities: dict[str, np.ndarray] = {}
    matrices: dict[int, np.ndarray] = {}
    for cutoff in config["cutoffs"]:
        matrix = np.zeros((3 * cutoff, 3 * cutoff), dtype=np.complex128)
        matrix[0, 0] = 1.0
        matrices[cutoff] = matrix
        for scenario in config["scenario_names"]:
            for backend in ("A", "B"):
                for position in range(72):
                    state = config["logical_state_schedule"][position % 6]
                    for round_index in range(12):
                        row_id = (
                            f"synthetic|fault|c{cutoff}|{scenario}|{backend}|"
                            f"p{position}|r{round_index}"
                        )
                        terminal = round_index == 11
                        rows.append(
                            {
                                "row_id": row_id,
                                "layer": "fault",
                                "cutoff": cutoff,
                                "scenario": scenario,
                                "backend": backend,
                                "logical_label": state,
                                "seed_position": position,
                                "round_index": round_index,
                                "terminal_round": terminal,
                                "mean_photon": 0.0,
                                "level_g": 1.0,
                                "level_e": 0.0,
                                "level_f": 0.0,
                                "logical_survival": 1.0,
                                "density_quantization_trace_distance_bound": (
                                    0.0 if terminal else None
                                ),
                            }
                        )
                        if terminal:
                            densities[row_id] = matrix
        for backend in ("A", "B"):
            for position in range(72):
                row_id = f"synthetic|shared|c{cutoff}|{backend}|p{position}"
                rows.append(
                    {
                        "row_id": row_id,
                        "layer": "shared",
                        "cutoff": cutoff,
                        "scenario": "",
                        "backend": backend,
                        "logical_label": "",
                        "seed_position": position,
                        "round_index": 0,
                        "terminal_round": True,
                        "mean_photon": 0.0,
                        "level_g": 1.0,
                        "level_e": 0.0,
                        "level_f": 0.0,
                        "logical_survival": 1.0,
                        "density_quantization_trace_distance_bound": 0.0,
                        "initial_state": "vacuum_f",
                        "action": "RESET",
                        "convergence_role": ("rao_blackwell_expected_reset_repair"),
                    }
                )
                densities[row_id] = matrices[cutoff]
    gates = diagnostic.evaluate(config, rows, densities)
    independently_recomputed = verifier._recompute(config, rows, densities)
    gate_ids = {str(gate["gate_id"]) for gate in gates}
    assert len(gates) == 1_454
    assert set(independently_recomputed) == gate_ids
    assert all(independently_recomputed[gate_id]["passed"] for gate_id in gate_ids)
    assert all(gate["passed"] for gate in gates)
    assert any("36->40" in gate_id for gate_id in gate_ids)
    assert any("40->44" in gate_id for gate_id in gate_ids)
    assert any("/c44/" in gate_id for gate_id in gate_ids)
    assert not any("28->" in gate_id or "32->" in gate_id for gate_id in gate_ids)
    assert not any("/c36/" in gate_id and "/tail/" in gate_id for gate_id in gate_ids)
