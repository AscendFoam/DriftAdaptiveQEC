from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.hybrid_state_output_validation import (
    HybridStateValidationConfig,
    _hmm_from_mapping,
    _implementation_sha256,
)
from cnn_fpga.decoder.hybrid_state_output import FORBIDDEN_DIRECT_OUTPUT_TOKENS
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES


ARTIFACT = Path("docs/t4_1_3_hybrid_state_output_validation.json")
SOURCE = Path("docs/t4_1_3_hybrid_state_output_source_data.csv")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"seeds": (1, 2, 3, 4, 5)}, "at least six"),
        ({"cycles_per_seed": 511}, "at least 512"),
        ({"history_cycles": 65, "output_stride_cycles": 32}, "divisible"),
        ({"nominal_seed_count": 8}, "leave both"),
        ({"bootstrap_replicates": 31}, "at least 32"),
    ],
)
def test_validation_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        HybridStateValidationConfig(**kwargs)


def test_hmm_mapping_loader_rejects_incomplete_checkpoint() -> None:
    with pytest.raises(ValueError, match="missing"):
        _hmm_from_mapping({"transition_matrix": []})


def test_committed_validation_artifact_passes_every_gate() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["gate_summary"]["passed"] == 17
    assert payload["gate_summary"]["failed"] == 0
    assert all(payload["gate_summary"]["gates"].values())
    assert payload["aggregate"]["outputs"] == 8 * 57
    assert payload["aggregate"]["lane_counts"] == {"nominal": 228, "stress": 228}


def test_checkpoint_and_implementation_hashes_are_current() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    checkpoint = Path(payload["registered_hmm_checkpoint"]["path"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == payload["registered_hmm_checkpoint"]["sha256"]
    assert _implementation_sha256() == payload["implementation_sha256"]


def test_source_data_is_complete_future_only_and_has_no_direct_action_or_truth_field() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 456
    fields = tuple(rows[0])
    normalized = ["".join(character for character in field.lower() if character.isalnum()) for field in fields]
    assert not any(token in field for field in normalized for token in FORBIDDEN_DIRECT_OUTPUT_TOKENS)
    assert not any(
        token in field
        for field in normalized
        for token in ("truth", "hidden", "logical", "oracle", "teacher", "label")
    )
    for row in rows:
        assert int(row["valid_from_cycle"]) == int(row["as_of_cycle"]) + 1
        assert int(row["expires_after_cycle"]) >= int(row["valid_from_cycle"])
        assert sum(float(row[f"regime_p_{name}"]) for name in REGIME_CLASSES) == pytest.approx(1.0)
        assert sum(float(row[f"recovery_depth_p_{index}"]) for index in range(7)) == pytest.approx(1.0)
        assert float(row["uncertainty_min_eigenvalue"]) >= -1.0e-10


def test_source_data_exercises_stage_hold_all_profiles_and_atomic_counts() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    action_counts = {
        action: sum(row["bank_action"] == action for row in rows)
        for action in ("stage_candidate", "hold_active")
    }
    mode_counts = {
        mode: sum(row["recommended_mode"] == mode for row in rows)
        for mode in ("normal", "x_recovery", "z_recovery", "leakage_hold", "fallback")
    }
    assert action_counts == payload["aggregate"]["bank_action_counts"]
    assert min(action_counts.values()) > 0
    assert mode_counts == payload["aggregate"]["recommended_mode_counts"]
    assert min(mode_counts.values()) > 0
    assert payload["aggregate"]["atomic_commits"] == action_counts["stage_candidate"]
    assert payload["aggregate"]["max_commit_parameter_error"] <= 1.0e-15


def test_claim_boundary_keeps_recovery_risk_and_hardware_bounded() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert "exact hidden recovery depth" in payload["claim_boundary"]["forbidden"]
    assert "RTL" in payload["claim_boundary"]["forbidden"]
    assert payload["output_schema"]["hardware_measured"] is False
    assert "observed-data recovery-burden posterior" in payload["output_schema"]["recovery_semantics"]
