from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import puviani_paper_constrained_artifacts as subject
from physics.fock_logical_channel import STATE_LABELS, logical_eigenstate_density


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase9/t9_1_3_puviani_paper_constrained.json"


def _pilot_config() -> dict[str, object]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    training = payload["training"]
    qualification = payload["six_state_qualification"]
    training.update(
        {
            "epochs": 1,
            "epochs_by_strategy": {"mf": 1, "nmf": 1},
            "full_cycles": 2,
            "cutoff": 4,
            "confirmation_cutoff": 4,
            "device": "cpu",
            "paired_root_seeds": [991_001],
            "validation_seeds": [992_003],
            "validation_interval": 1,
            "mf_batch_size": 1,
            "nmf_batch_size": 1,
            "validation_batch_size": 1,
            "seed_derivation": {
                "training_rollout_seed": "paired_root_seed*1000003+epoch",
                "paired_across_mf_and_nmf": True,
                "root_seed_count": 1,
                "strategy_count": 2,
                "independently_trained_agent_count": 2,
            },
        }
    )
    qualification.update(
        {
            "seeds": [993_007],
            "confirmation_seeds": [994_009],
            "batch_size": 1,
            "confirmation_batch_size": 1,
            "full_cycles": 2,
            "primary_cutoff": 4,
            "confirmation_cutoff": 4,
        }
    )
    return payload


@pytest.fixture(scope="module")
def pilot_evidence() -> tuple[
    dict[str, object],
    dict[str, tuple[object, dict[str, int]]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    pytest.importorskip("torch")
    config = _pilot_config()
    seed = int(config["training"]["paired_root_seeds"][0])
    selected = {
        strategy: (
            subject.build_policy(
                strategy, subject._strategy_config(config, strategy), seed
            ),
            {"root_seed": seed},
        )
        for strategy in ("mf", "nmf")
    }
    trajectories, events, _ = subject._evaluate_six_state(config, selected)
    return config, selected, trajectories, events


def _write_matrix_and_metrics(row: dict[str, object], matrix: np.ndarray) -> None:
    row.update(
        {
            "projected_rho_00_real": float(matrix[0, 0].real),
            "projected_rho_00_imag": float(matrix[0, 0].imag),
            "projected_rho_01_real": float(matrix[0, 1].real),
            "projected_rho_01_imag": float(matrix[0, 1].imag),
            "projected_rho_10_real": float(matrix[1, 0].real),
            "projected_rho_10_imag": float(matrix[1, 0].imag),
            "projected_rho_11_real": float(matrix[1, 1].real),
            "projected_rho_11_imag": float(matrix[1, 1].imag),
        }
    )
    metrics = subject._projected_logical_metrics(matrix, str(row["initial_state"]))
    for name in (
        "code_survival",
        "code_weighted_I",
        "code_weighted_X",
        "code_weighted_Y",
        "code_weighted_Z",
        "state_fidelity_unconditional",
        "signed_axis_signal",
    ):
        row[name] = float(metrics[name])


def _replace_action(
    event: dict[str, object], prefix: str, value: np.ndarray
) -> None:
    project_key = f"{prefix}project_action_order_json"
    gqf_key = f"{prefix}gqf_action_order_json"
    hash_key = f"{prefix}project_action_sha256"
    event[project_key] = json.dumps(value.tolist(), separators=(",", ":"))
    event[gqf_key] = json.dumps(
        subject.project_to_gqf_action(value).tolist(), separators=(",", ":")
    )
    event[hash_key] = hashlib.sha256(
        np.ascontiguousarray(value).tobytes()
    ).hexdigest()


def test_explicit_six_state_vectors_match_frozen_logical_densities() -> None:
    for label in STATE_LABELS:
        vector = subject._logical_state_vector(label)
        assert vector.shape == (2,)
        assert np.vdot(vector, vector) == pytest.approx(1.0, abs=3.0e-16)
        assert np.allclose(
            np.outer(vector, vector.conj()),
            logical_eigenstate_density(label),
            rtol=0.0,
            atol=2.0e-16,
        )


def test_projected_metrics_are_unconditional_and_reject_nonphysical_rows() -> None:
    survival = 0.73
    matrix = survival * logical_eigenstate_density("y_minus")
    metrics = subject._projected_logical_metrics(matrix, "y_minus")
    assert metrics["code_survival"] == pytest.approx(survival, abs=1.0e-15)
    assert metrics["state_fidelity_unconditional"] == pytest.approx(
        survival, abs=1.0e-15
    )
    assert metrics["signed_axis_signal"] == pytest.approx(survival, abs=1.0e-15)
    assert metrics["code_weighted_Y"] == pytest.approx(-survival, abs=1.0e-15)

    with pytest.raises(ValueError, match="not Hermitian"):
        subject._projected_logical_metrics(
            np.asarray([[0.5, 0.2], [0.0, 0.5]], dtype=np.complex128),
            "z_plus",
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        subject._projected_logical_metrics(
            np.asarray([[1.1, 0.0], [0.0, -0.1]], dtype=np.complex128),
            "z_plus",
        )


def test_same_backend_replay_binds_all_rows_to_selected_models_rng_and_physics(
    pilot_evidence: tuple[
        dict[str, object],
        dict[str, tuple[object, dict[str, int]]],
        list[dict[str, object]],
        list[dict[str, object]],
    ],
) -> None:
    config, selected, trajectories, events = pilot_evidence
    audit = subject._six_state_raw_replay_audit(
        config, selected, trajectories, events
    )
    assert audit["all_rows_replayed"] is True
    assert audit["trajectory_count"] == len(trajectories) == 36
    assert audit["event_count"] == len(events) == 144
    assert audit["group_count"] == 36
    assert subject._six_state_raw_replay_audit_valid(
        config,
        audit,
        trajectory_count=len(trajectories),
        event_count=len(events),
    )


@pytest.mark.parametrize(
    "mutation",
    ("rho_and_metrics", "probability", "outcome", "coordinated_action", "winner"),
)
def test_raw_replay_rejects_coordinated_internal_consistency_forgeries(
    pilot_evidence: tuple[
        dict[str, object],
        dict[str, tuple[object, dict[str, int]]],
        list[dict[str, object]],
        list[dict[str, object]],
    ],
    mutation: str,
) -> None:
    config, selected, original_trajectories, original_events = pilot_evidence
    trajectories = deepcopy(original_trajectories)
    events = deepcopy(original_events)
    if mutation == "rho_and_metrics":
        row = trajectories[0]
        matrix = np.asarray(
            [
                [
                    complex(row["projected_rho_00_real"], row["projected_rho_00_imag"]),
                    complex(row["projected_rho_01_real"], row["projected_rho_01_imag"]),
                ],
                [
                    complex(row["projected_rho_10_real"], row["projected_rho_10_imag"]),
                    complex(row["projected_rho_11_real"], row["projected_rho_11_imag"]),
                ],
            ],
            dtype=np.complex128,
        )
        _write_matrix_and_metrics(row, 0.99 * matrix)
    elif mutation == "probability":
        event = events[0]
        event["selected_branch_probability"] = min(
            0.999999,
            max(1.0e-9, float(event["selected_branch_probability"]) * 0.99),
        )
    elif mutation == "outcome":
        events[0]["outcome_g0_e1"] = 1 - int(events[0]["outcome_g0_e1"])
    elif mutation == "coordinated_action":
        learned = next(row for row in trajectories if row["strategy"] == "mf")
        matching = sorted(
            [row for row in events if row["trajectory_id"] == learned["trajectory_id"]],
            key=lambda row: int(row["half_cycle"]),
        )
        value = np.asarray(
            json.loads(matching[0]["post_measurement_project_action_order_json"]),
            dtype=np.float64,
        )
        value[0] += 1.0e-6
        _replace_action(matching[0], "post_measurement_", value)
        _replace_action(matching[1], "", value)
        assert subject._event_timeline_valid(events, trajectories, half_cycles=4)
    elif mutation == "winner":
        row = next(row for row in trajectories if row["strategy"] == "nmf")
        row["selected_agent_seed"] = int(row["selected_agent_seed"]) + 1
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        subject._six_state_raw_replay_audit(
            config, selected, trajectories, events
        )


def test_event_timeline_rejects_absolute_instead_of_residual_bound(
    pilot_evidence: tuple[
        dict[str, object],
        dict[str, tuple[object, dict[str, int]]],
        list[dict[str, object]],
        list[dict[str, object]],
    ],
) -> None:
    _, _, trajectories, original_events = pilot_evidence
    events = deepcopy(original_events)
    learned = next(row for row in trajectories if row["strategy"] == "mf")
    matching = sorted(
        [row for row in events if row["trajectory_id"] == learned["trajectory_id"]],
        key=lambda row: int(row["half_cycle"]),
    )
    nominal = np.asarray(
        json.loads(matching[0]["post_measurement_project_action_order_json"]),
        dtype=np.float64,
    )
    nominal[0] = float(subject.nominal_sbs_parameters(device="cpu")[0]) + 2.01
    _replace_action(matching[0], "post_measurement_", nominal)
    _replace_action(matching[1], "", nominal)
    assert not subject._event_timeline_valid(events, trajectories, half_cycles=4)
