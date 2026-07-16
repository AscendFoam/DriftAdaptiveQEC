from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.fast_path_fixed_point_validation import (
    _implementation_sha256,
    _profile_images,
    _row_hash,
    _simulate_ler,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "t4_2_4_fast_path_fixed_point_validation.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_production_artifact_passes_all_gates_and_is_source_bound(
    artifact: dict,
) -> None:
    assert artifact["task_id"] == "T4.2.4"
    assert artifact["status"] == "PASS"
    assert artifact["implementation_sha256"] == _implementation_sha256()
    assert artifact["gate_summary"]["passed"] == 21
    assert artifact["gate_summary"]["failed"] == 0
    assert all(artifact["gate_summary"]["gates"].values())


def test_selected_word_contract_and_representation_counts_are_exact(artifact: dict) -> None:
    contract = artifact["selected_contract"]
    assert contract["adc_bits"] == 10
    assert contract["address_bits"] == 8
    assert contract["interpolation_fraction_bits"] == 2
    assert contract["llr_integer_bits"] == 9
    assert contract["llr_fractional_bits"] == 12
    proxy = artifact["representation_proxy"]["selected_p10_a8_q9_12"]
    assert proxy["rom_bits_per_bank"] == 11_308
    assert proxy["double_bank_rom_bits"] == 22_616
    assert proxy["registered_eight_bank_artifact_bits"] == 90_464
    assert proxy["live_event_state_bits"] == 55
    assert proxy["live_health_state_and_input_bits"] == 182
    assert len(artifact["arithmetic_rules"]) == 9
    assert "ties-to-even" in artifact["arithmetic_rules"]["lut_interpolation"]
    assert "no wrap" in artifact["arithmetic_rules"]["event_health_counters"]


def test_exhaustive_source_data_has_all_87040_codes_and_zero_selected_mismatch(
    artifact: dict,
) -> None:
    source = artifact["source_data"]["exhaustive_codes"]
    rows = _rows(ROOT / source["path"])
    assert len(rows) == source["rows"] == 87_040
    assert {row["profile"] for row in rows} == set(artifact["profile_configs"])
    selected = artifact["exhaustive_code_summary"]["selected_p10_a8_q9_12"]
    assert selected["rows"] == 16_384
    assert selected["hard_action_mismatch_count"] == 0
    assert selected["mean_abs_llr_value_error"] == pytest.approx(9.466707706451416e-05)
    assert selected["max_abs_llr_value_error"] == pytest.approx(0.0048828125)


def test_ler_source_data_is_paired_across_profiles_banks_and_seeds(artifact: dict) -> None:
    source = artifact["source_data"]["ler"]
    rows = _rows(ROOT / source["path"])
    assert len(rows) == source["rows"] == 128
    keys = {(row["profile"], row["active_bank_version"], row["seed"]) for row in rows}
    assert len(keys) == len(rows)
    assert {int(row["active_bank_version"]) for row in rows} == set(range(8))
    assert {int(row["seed"]) for row in rows} == {42401, 42403, 42407, 42409}
    assert all(int(row["samples"]) == 2048 for row in rows)
    assert all(int(row["fallback_count"]) == 0 for row in rows)
    assert all(int(row["map_accepted_count"]) == 2048 for row in rows)


def test_selected_ler_impact_is_small_but_reported_with_paired_ci(artifact: dict) -> None:
    selected = artifact["ler_summary"]["selected_p10_a8_q9_12"]
    effect = selected["paired_quantized_minus_float"]
    assert effect["clusters"] == 32
    assert effect["mean"] == pytest.approx(3.0517578125e-05)
    assert effect["ci_low"] == pytest.approx(-4.57763671875e-05)
    assert effect["ci_high"] == pytest.approx(0.0001220703125)
    assert selected["action_disagreement_mean"] == pytest.approx(9.1552734375e-05)
    assert effect["ci_low"] <= 0.0 <= effect["ci_high"]


def test_dense_and_low_profiles_preserve_precision_counterevidence(artifact: dict) -> None:
    summaries = artifact["ler_summary"]
    dense = summaries["dense_p12_a10_q10_14"]
    selected = summaries["selected_p10_a8_q9_12"]
    low = summaries["low_p6_a4_q5_6"]
    assert dense["action_disagreement_mean"] == 0.0
    assert low["action_disagreement_mean"] > selected["action_disagreement_mean"]
    assert low["paired_quantized_minus_float"]["ci_low"] < 0.0
    assert low["paired_quantized_minus_float"]["ci_high"] > 0.0


def test_negative_and_hardware_claim_boundaries_are_fail_closed(artifact: dict) -> None:
    assert artifact["negative_audit"] and all(artifact["negative_audit"].values())
    assert artifact["online_integer_audit"]["integer_only"] is True
    hardware = artifact["hardware_fields"]
    for field in (
        "target_lut_count",
        "target_ff_count",
        "target_bram_count",
        "target_dsp_count",
        "fmax_mhz",
    ):
        assert hardware[field] is None
    assert hardware["rtl_measured"] is False and hardware["board_measured"] is False
    assert "device-calibrated" in artifact["claim_boundary"]["forbidden"]


def test_independent_selected_profile_replay_is_bit_deterministic() -> None:
    images = _profile_images()["selected_p10_a8_q9_12"]
    first, first_timing = _simulate_ler(
        {"selected_p10_a8_q9_12": images}, samples_per_run=1024
    )
    second, second_timing = _simulate_ler(
        {"selected_p10_a8_q9_12": images}, samples_per_run=1024
    )
    assert _row_hash(first) == _row_hash(second)
    assert first_timing == second_timing
