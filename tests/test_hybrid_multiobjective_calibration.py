import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.hybrid_multiobjective_calibration import (
    HybridMultiObjectiveValidationConfig,
    _implementation_sha256,
    build_hybrid_multiobjective_validation,
)
from cnn_fpga.decoder.hybrid_multiobjective import OBJECTIVE_NAMES
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES


JSON_PATH = Path("docs/t4_1_4_hybrid_multiobjective_validation.json")
CSV_PATH = Path("docs/t4_1_4_hybrid_multiobjective_source_data.csv")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"future_horizon_cycles": 7}, "at least"),
        ({"cycles_per_seed": 2050}, "divisible"),
        ({"training_seed_count": 7}, "leave evaluation"),
        ({"minimum_unsafe_recall": 0.0}, "unsafe_recall"),
    ],
)
def test_validation_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HybridMultiObjectiveValidationConfig(**kwargs)


def test_committed_artifact_passes_every_gate_and_covers_all_objectives() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["gate_summary"]["passed"] == 19
    assert payload["gate_summary"]["failed"] == 0
    assert all(payload["gate_summary"]["gates"].values())
    assert set(payload["evaluation_frozen"]["raw_objectives"]) == set(OBJECTIVE_NAMES)
    assert set(payload["evaluation_frozen"]["leave_one_objective_out"]) == set(OBJECTIVE_NAMES)


def test_artifact_and_input_hashes_are_current() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert payload["implementation_sha256"] == _implementation_sha256()
    for key in ("csv", "manifest"):
        entry = payload["input_source"]
        path = Path(entry[f"{key}_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry[f"{key}_sha256"]


def test_source_data_has_exact_future_alignment_and_explicit_offline_scope() -> None:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 448
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert {row["split"] for row in rows} == {"training", "validation", "evaluation"}
    assert all(int(row["future_start_cycle"]) == int(row["as_of_cycle"]) + 1 for row in rows)
    assert all(
        int(row["future_end_cycle_exclusive"]) - int(row["future_start_cycle"]) == 32
        for row in rows
    )
    assert all(row["scope"] == "offline_future_aligned_calibration_record" for row in rows)
    target_fields = [field for field in rows[0] if field.startswith("offline_future_target_")]
    assert len(target_fields) == 9


def test_strict_split_and_all_regime_labels_are_nonempty() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert payload["split_counts"] == {"training": 168, "validation": 112, "evaluation": 168}
    seed_sets = [set(payload["split_seeds"][split]) for split in ("training", "validation", "evaluation")]
    assert [len(values) for values in seed_sets] == [3, 2, 3]
    assert not seed_sets[0] & seed_sets[1]
    assert not seed_sets[0] & seed_sets[2]
    assert not seed_sets[1] & seed_sets[2]
    for counts in payload["regime_target_counts"].values():
        assert set(counts) == set(REGIME_CLASSES)
        assert min(counts.values()) > 0


def test_proper_calibration_improves_validation_without_evaluation_selection() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    identity = payload["validation_identity"]
    calibrated = payload["validation_calibrated"]
    assert calibrated["diagnostics"]["regime_nll"] < identity["diagnostics"]["regime_nll"]
    assert (
        calibrated["raw_objectives"]["uncertainty_calibration"]
        < identity["raw_objectives"]["uncertainty_calibration"]
    )
    assert calibrated["diagnostics"]["required_fallback_recall"] >= 0.90
    assert payload["evaluation_frozen"]["selection_provenance"]["evaluation_used_for_selection"] is False


def test_negative_fallback_result_is_preserved_not_hidden() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    diagnostics = payload["evaluation_frozen"]["diagnostics"]
    assert diagnostics["required_fallback_recall"] == 1.0
    assert diagnostics["false_fallback_rate"] == 1.0
    assert payload["evaluation_frozen"]["raw_objectives"]["false_fallback"] == 1.0
    assert 0 < payload["required_fallback_counts"]["evaluation"] < payload["split_counts"]["evaluation"]


def test_rebuild_is_deterministic_for_current_registered_sources() -> None:
    committed = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rebuilt, rows = build_hybrid_multiobjective_validation()
    assert rebuilt["status"] == "PASS"
    assert rebuilt["source_rows_sha256"] == committed["source_rows_sha256"]
    assert rebuilt["calibration_manifest"] == committed["calibration_manifest"]
    assert len(rows) == 448
