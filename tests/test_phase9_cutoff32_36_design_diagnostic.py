from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_cutoff32_36_design_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[1]
STATES = ("0", "1", "+", "-", "+i", "-i")
SCENARIOS = ("step", "telegraph", "burst", "compound")


def _config() -> dict:
    return json.loads(
        (
            ROOT
            / "configs/phase9/t_risk_20260728_01_cutoff32_36_design_extension.json"
        ).read_text(encoding="utf-8")
    )


def _synthetic_evidence(
    *, cutoff36_mean_photon_shift: float = 0.0
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    densities: dict[str, np.ndarray] = {}
    vacuum = {}
    for cutoff in (28, 32, 36):
        density = np.zeros((3 * cutoff, 3 * cutoff), dtype=np.complex128)
        density[0, 0] = 1.0
        vacuum[cutoff] = density
    for cutoff in (28, 32, 36):
        for scenario in SCENARIOS:
            for backend in ("A", "B"):
                for position in range(72):
                    state = STATES[position % 6]
                    for round_index in range(12):
                        terminal = round_index == 11
                        row_id = (
                            f"fault/c{cutoff}/{scenario}/{backend}/"
                            f"p{position}/r{round_index}"
                        )
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
                                "initial_state": "",
                                "action": "IDLE",
                                "mean_photon": (
                                    cutoff36_mean_photon_shift
                                    if cutoff == 36
                                    else 0.0
                                ),
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
                            densities[row_id] = vacuum[cutoff]
        for backend in ("A", "B"):
            for position in range(72):
                row_id = f"shared/c{cutoff}/{backend}/p{position}"
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
                        "initial_state": "vacuum_f",
                        "action": "RESET",
                        "mean_photon": 0.0,
                        "level_g": 1.0,
                        "level_e": 0.0,
                        "level_f": 0.0,
                        "logical_survival": 1.0,
                        "density_quantization_trace_distance_bound": 0.0,
                    }
                )
                densities[row_id] = vacuum[cutoff]
    assert len(rows) == 21168
    assert len(densities) == 2160
    return rows, densities


def test_full_1454_gate_fixture_passes_without_cross_state_averaging() -> None:
    rows, densities = _synthetic_evidence()
    gates = diagnostic.evaluate(_config(), rows, densities)
    assert len(gates) == 1454
    assert len({gate["gate_id"] for gate in gates}) == 1454
    assert all(gate["passed"] for gate in gates)
    assert {
        gate["logical_state"]
        for gate in gates
        if gate["family"].startswith("fault_")
    } == set(STATES)
    assert {
        gate["family"] for gate in gates
    } == {
        "fault_density",
        "fault_scalar",
        "fault_absolute_tail",
        "shared_density",
        "shared_scalar",
        "shared_absolute_tail",
    }


def test_registered_scalar_margin_failure_produces_no_go_rows() -> None:
    rows, densities = _synthetic_evidence(cutoff36_mean_photon_shift=0.2)
    gates = diagnostic.evaluate(_config(), rows, densities)
    failed = [gate for gate in gates if not gate["passed"]]
    assert failed
    assert all(
        gate["family"] == "fault_scalar"
        and gate["metric"] == "mean_photon"
        and gate["cutoff_or_increment"] == "32->36"
        for gate in failed
    )


def test_missing_round_and_duplicate_row_fail_closed() -> None:
    rows, densities = _synthetic_evidence()
    with pytest.raises(RuntimeError, match="round coverage"):
        diagnostic.evaluate(_config(), rows[1:], densities)

    duplicated = [*rows, deepcopy(rows[0])]
    with pytest.raises(RuntimeError, match="round coverage"):
        diagnostic.evaluate(_config(), duplicated, densities)


def test_quantization_certificate_votes_in_conservative_density_point() -> None:
    gate = diagnostic._gate(
        gate_id="fixture",
        family="fault_density",
        contrast="within_backend_cutoff",
        scenario="step",
        state="0",
        stage="terminal",
        metric="density_trace_distance",
        cutoff="32->36",
        backend="A",
        estimate=0.07,
        quantization=0.006,
        margin=0.075,
        clusters=12,
    )
    assert gate["estimate"] == 0.07
    assert gate["conservative_point"] == pytest.approx(0.076)
    assert gate["passed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "twin_qualification",
        "ler",
        "lifetime",
        "physical_break_even",
        "official_puviani_exact",
        "puviani_nmf_surpass",
        "external_sota",
        "hardware_measured",
    ],
)
def test_design_diagnostic_claim_firewall_stays_null(field) -> None:
    assert diagnostic.CLAIM_BOUNDARY[field] is None
