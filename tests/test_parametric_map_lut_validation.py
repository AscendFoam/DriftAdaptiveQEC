from __future__ import annotations

import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark.parametric_map_lut_validation import (
    DEFAULT_CSV,
    DEFAULT_IMAGES,
    DEFAULT_JSON,
    ROOT,
    ValidationConfig,
    _implementation_sha256,
    run_validation,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_artifact_passes_all_twenty_registered_gates() -> None:
    payload = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "t4.2.1-parametric-map-lut-validation-v1"
    assert payload["task_id"] == "T4.2.1"
    assert payload["status"] == "PASS"
    assert payload["gate_summary"]["passed"] == 20
    assert payload["gate_summary"]["failed"] == 0
    assert all(payload["gate_summary"]["gates"].values())
    assert payload["implementation_sha256"] == _implementation_sha256()


def test_production_source_data_is_hash_bound_and_exhaustive() -> None:
    payload = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    assert payload["source_data"]["rows"] == 8 * 2 * 1024
    assert payload["source_data"]["sha256"] == _sha256(DEFAULT_CSV)
    with DEFAULT_CSV.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16_384
    assert {int(row["active_bank_version"]) for row in rows} == set(range(8))
    assert {int(row["phase_bit"]) for row in rows} == {0, 1}
    assert {int(row["syndrome_code"]) for row in rows} == set(range(1024))
    assert sum(int(row["action_mismatch"]) for row in rows) == 0


def test_bank_image_artifact_has_full_tables_unique_hashes_and_live_hash() -> None:
    payload = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    images = json.loads(DEFAULT_IMAGES.read_text(encoding="utf-8"))["images"]
    assert payload["image_artifact"]["sha256"] == _sha256(DEFAULT_IMAGES)
    assert len(images) == 8
    assert [image["active_bank_version"] for image in images] == list(range(8))
    assert len({image["source_params_sha256"] for image in images}) == 8
    assert len({image["image_sha256"] for image in images}) == 8
    assert all(len(image["table_codes"]) == 2 for image in images)
    assert all(len(table) == 257 for image in images for table in image["table_codes"])


def test_selected_metrics_preserve_actions_and_interpolation_beats_nearest() -> None:
    metrics = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))["selected_metrics"]
    assert metrics["action_mismatch_count"] == 0
    assert metrics["action_mismatch_rate"] == 0.0
    assert metrics["mean_abs_llr_code_error"] < 0.4
    assert metrics["max_abs_llr_code_error"] <= 20
    assert metrics["interpolation_to_nearest_error_ratio"] < 0.005
    assert metrics["phase_table_difference_count"] > 2000


def test_address_convergence_is_strict_and_action_stable() -> None:
    rows = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))["convergence"]
    assert [row["address_bits"] for row in rows] == [5, 6, 7, 8]
    errors = [row["mean_abs_llr_code_error"] for row in rows]
    assert all(left > right for left, right in zip(errors, errors[1:]))
    assert all(row["action_mismatch_count"] == 0 for row in rows)


def test_pipeline_and_resource_claims_stay_contract_only() -> None:
    payload = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    pipeline = payload["pipeline_audit"]
    resource = payload["resource_contract"]
    assert pipeline["latencies"] == [5, 5]
    assert pipeline["ii_one_output_count"] == 16
    assert pipeline["old_inflight_image_latched"] is True
    assert resource["dual_bank_table_bits"] == 22_616
    assert resource["runtime_dividers"] == 0
    assert resource["runtime_exp_log_units"] == 0
    for field in (
        "target_lut_count",
        "target_ff_count",
        "target_bram_count",
        "target_dsp_count",
        "fmax_mhz",
    ):
        assert resource[field] is None
    assert resource["rtl_measured"] is False
    assert resource["board_measured"] is False


def test_claim_boundary_rejects_joint_map_and_hardware_upgrade() -> None:
    boundary = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))["claim_boundary"]
    assert "axis-marginal" in boundary["allowed"]
    for phrase in (
        "full correlated 2D MAP optimality",
        "event-FSM/frame integration",
        "RTL synthesis",
        "FPGA timing",
        "board/device measurement",
    ):
        assert phrase in boundary["forbidden"]


def test_reduced_replay_regenerates_complete_deterministic_artifacts(tmp_path) -> None:
    first_json = tmp_path / "first.json"
    first_csv = tmp_path / "first.csv"
    first_images = tmp_path / "first_images.json"
    config = ValidationConfig(
        adc_bits=8,
        selected_address_bits=8,
        convergence_address_bits=(5, 6, 7, 8),
    )
    payload = run_validation(
        config,
        json_path=first_json,
        csv_path=first_csv,
        images_path=first_images,
    )
    assert payload["status"] == "PASS"
    assert payload["gate_summary"]["failed"] == 0
    assert payload["source_data"]["rows"] == 8 * 2 * 256
    assert payload["source_data"]["sha256"] == _sha256(first_csv)
    assert payload["image_artifact"]["sha256"] == _sha256(first_images)
    assert first_json.exists()


def test_artifact_paths_are_registered_under_docs() -> None:
    for path in (DEFAULT_JSON, DEFAULT_CSV, DEFAULT_IMAGES):
        assert path.is_file()
        assert path.parent == ROOT / "docs"
