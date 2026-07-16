from __future__ import annotations

import hashlib
import json

import pytest

from cnn_fpga.benchmark.experimental_history_validation import (
    ExperimentalHistoryValidationConfig,
    build_experimental_history_validation,
)
from cnn_fpga.data.experimental_history import FEATURE_NAMES, FORBIDDEN_INPUT_TOKENS, UPDATE_STATUSES
from cnn_fpga.runtime.run_length_fsm import FSM_MODES


@pytest.fixture(scope="module")
def compact_validation() -> tuple[dict[str, object], list[dict[str, object]]]:
    return build_experimental_history_validation(
        ExperimentalHistoryValidationConfig(
            seeds=(20261301, 20261302, 20261303, 20261304, 20261305, 20261306),
            cycles_per_seed=1024,
            history_cycles=64,
            llr_clip=3.0,
            run_length_clip=2,
            bank_version_clip=7,
            pending_window_clip=1,
        )
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"seeds": (1, 2, 3, 4, 5)}, "at least six"),
        ({"seeds": (1, 2, 3, 4, 5, 5)}, "at least six"),
        ({"cycles_per_seed": 1023}, "at least 1024"),
        ({"history_cycles": 31}, "lie in"),
        ({"llr_clip": 0.0}, "positive"),
        ({"run_length_clip": True}, "positive integer"),
    ],
)
def test_validation_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ExperimentalHistoryValidationConfig(**kwargs)


def test_compact_production_replay_passes_every_gate(compact_validation: object) -> None:
    payload, rows = compact_validation  # type: ignore[misc]
    assert payload["status"] == "PASS"
    gate_summary = payload["gate_summary"]
    assert gate_summary["failed"] == 0  # type: ignore[index]
    assert gate_summary["passed"] == 17  # type: ignore[index]
    assert len(rows) == 6 * 1024


def test_source_rows_are_exactly_aligned_and_deployable(compact_validation: object) -> None:
    _, rows = compact_validation  # type: ignore[misc]
    expected_fields = {
        "seed",
        "cycle",
        "history_valid_cycles",
        "history_start_cycle",
        "history_end_cycle",
        "update_status",
        "action_mode",
        "scheduler_event_kinds",
        *FEATURE_NAMES,
    }
    assert set(rows[0]) == expected_fields
    for row in rows:
        cycle = int(row["cycle"])
        assert int(row["history_end_cycle"]) == cycle
        assert int(row["history_start_cycle"]) == max(0, cycle - 63)
        assert int(row["history_valid_cycles"]) == min(cycle + 1, 64)
    normalized = ["".join(character for character in field.lower() if character.isalnum()) for field in rows[0]]
    assert not any(token in field for field in normalized for token in FORBIDDEN_INPUT_TOKENS)


def test_replay_exercises_every_required_categorical_path(compact_validation: object) -> None:
    payload, _ = compact_validation  # type: ignore[misc]
    aggregate = payload["aggregate"]
    assert set(aggregate["update_status_counts"]) == set(UPDATE_STATUSES)  # type: ignore[index]
    assert min(aggregate["update_status_counts"].values()) > 0  # type: ignore[index,union-attr]
    assert set(aggregate["action_mode_counts"]) == set(FSM_MODES)  # type: ignore[index]
    assert min(aggregate["action_mode_counts"].values()) > 0  # type: ignore[index,union-attr]
    assert min(aggregate["observed_outcome_counts"].values()) > 0  # type: ignore[index,union-attr]


def test_source_hash_is_recomputable_from_rows(compact_validation: object) -> None:
    payload, rows = compact_validation  # type: ignore[misc]
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    assert digest.hexdigest() == payload["source_rows_sha256"]
    assert len(payload["implementation_sha256"]) == 64


def test_committed_artifact_preserves_full_workload_and_claim_boundary() -> None:
    payload = json.loads(
        open("docs/t4_1_2_experimental_history_validation.json", encoding="utf-8").read()
    )
    assert payload["status"] == "PASS"
    assert payload["aggregate"]["source_data_rows"] == 8 * 2048
    assert payload["aggregate"]["feature_count"] == 53
    assert payload["gate_summary"]["failed"] == 0
    assert "device-calibrated" in payload["claim_boundary"]["forbidden"]
    assert payload["history_schema"]["hardware_measured"] is False
