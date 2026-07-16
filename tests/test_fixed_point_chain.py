from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.decoder.linear_runtime import FixedPointFormat
from cnn_fpga.runtime.fixed_point_chain import (
    BANK_FAULT_MODES,
    ROOT,
    FixedPointStressConfig,
    LUTParameterBank,
    PrecisionProfile,
    UniformCodeFormat,
    _build_bank_image,
    _flip_signed_bit,
    _physical_trace,
    _sha256_sources,
    _simulate_float_reference,
    base_profile,
    estimate_wrapped_state,
    precision_profiles,
    resource_proxy,
    simulate_quantized_profile,
)
from cnn_fpga.runtime.plot_fixed_point_chain import DEFAULT_FIGURE_AUDIT
from physics.constants import LATTICE_CONST


ARTIFACT = ROOT / "docs" / "t2_4_3_fixed_point_validation.json"
SOURCE_CSV = ROOT / "docs" / "t2_4_3_precision_resource_ler.csv"


def test_fixed_point_words_have_explicit_codes_ties_and_saturation() -> None:
    fmt = FixedPointFormat(integer_bits=1, fractional_bits=2)
    codes, saturated = fmt.encode(np.asarray([-3.0, -0.375, 0.375, 3.0]))
    assert fmt.word_bits == 4
    assert (fmt.min_code, fmt.max_code) == (-8, 7)
    np.testing.assert_array_equal(codes, [-8, -2, 2, 7])
    np.testing.assert_array_equal(saturated, [True, False, False, True])
    np.testing.assert_array_equal(fmt.decode(codes), [-2.0, -0.5, 0.5, 1.75])

    with pytest.raises(ValueError, match="finite"):
        fmt.encode([np.nan])
    with pytest.raises(ValueError, match="outside"):
        fmt.decode([8])
    with pytest.raises(TypeError, match="integers"):
        fmt.decode([1.0])
    with pytest.raises(ValueError, match="above 63"):
        FixedPointFormat(40, 30)


def test_uniform_code_uses_level_centres_and_marks_both_outside_edges() -> None:
    fmt = UniformCodeFormat(bits=2, minimum=-1.0, maximum=1.0)
    quantized, codes, saturated = fmt.quantize([-1.0, -0.5, 0.0, 0.999, 1.0])
    np.testing.assert_array_equal(codes, [0, 1, 2, 3, 3])
    np.testing.assert_allclose(quantized, [-0.75, -0.25, 0.25, 0.75, 0.75])
    np.testing.assert_array_equal(saturated, [False, False, False, False, True])
    with pytest.raises(ValueError, match="outside"):
        fmt.decode([4])


def test_signed_bit_flip_is_word_width_exact_and_reversible() -> None:
    values = np.asarray([-8, -1, 0, 7], dtype=np.int64)
    flipped = _flip_signed_bit(values, word_bits=4, bit_index=3)
    np.testing.assert_array_equal(flipped, [0, 7, -8, -1])
    np.testing.assert_array_equal(
        _flip_signed_bit(flipped, word_bits=4, bit_index=3), values
    )
    with pytest.raises(ValueError, match="outside"):
        _flip_signed_bit(np.asarray([8]), word_bits=4, bit_index=0)


def test_profiles_cover_every_required_axis_and_resources_are_exact_proxies() -> None:
    profiles = precision_profiles()
    axes: dict[str, list[PrecisionProfile]] = {}
    for profile in profiles:
        axes.setdefault(profile.curve_axis, []).append(profile)
    assert len(profiles) == 42
    for axis in (
        "adc_bits",
        "lut_address_bits",
        "llr_fractional_bits",
        "threshold_fractional_bits",
        "state_bits",
        "update_period_windows",
    ):
        assert len(axes[axis]) == 6
    assert len(axes["joint_precision"]) == 5

    profile = base_profile()
    resource = resource_proxy(profile, window_size=256)
    assert resource["dual_bank_lut_bits"] == 2 * 256 * 12
    assert resource["dual_bank_state_threshold_bits"] == 2 * (24 + 10)
    assert resource["total_dual_bank_storage_bits"] == 6212
    assert resource["replay_window_bits"] == 2560
    assert resource["mean_update_payload_bits_per_window"] == 3106
    assert resource["identity"] == "exact_representation_proxy_not_synthesis"
    assert resource["fpga_lut_count"] is None
    assert resource["bram_count"] is None
    assert resource["dsp_count"] is None


def test_physical_trace_is_seed_deterministic_and_parity_reconstructs() -> None:
    cfg = FixedPointStressConfig(
        n_samples=2048,
        window_size=128,
        seeds=(7, 11),
        bootstrap_replicates=1000,
    )
    first = _physical_trace(cfg, 7)
    second = _physical_trace(cfg, 7)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    indices = np.floor(first["raw_error"] / LATTICE_CONST + 0.5).astype(np.int64)
    np.testing.assert_array_equal(first["true_parity"], np.mod(indices, 2).astype(bool))
    reconstructed = first["syndrome"] + indices * LATTICE_CONST
    np.testing.assert_allclose(reconstructed, first["raw_error"], atol=1.0e-12)
    half = LATTICE_CONST / 2.0
    assert np.all(first["observed_syndrome"] >= -half)
    assert np.all(first["observed_syndrome"] < half)


def test_wrapped_state_estimator_recovers_observed_distribution_without_truth() -> None:
    rng = np.random.default_rng(243)
    mean_true = 0.48
    sigma_true = 0.25
    samples = mean_true + rng.normal(0.0, sigma_true, size=100_000)
    half = LATTICE_CONST / 2.0
    observed = np.mod(samples + half, LATTICE_CONST) - half
    mean_hat, sigma_hat = estimate_wrapped_state(observed)
    assert mean_hat == pytest.approx(mean_true, abs=0.004)
    assert sigma_hat == pytest.approx(sigma_true, abs=0.004)
    with pytest.raises(ValueError, match="at least four"):
        estimate_wrapped_state(np.asarray([0.0, 0.1, 0.2]))


def test_parameter_bank_faults_are_explicit_and_stale_commit_preserves_version() -> None:
    profile = PrecisionProfile(
        profile_id="unit",
        curve_axis="unit",
        axis_value=1.0,
        adc_bits=6,
        lut_address_bits=4,
        llr_fractional_bits=3,
        threshold_fractional_bits=3,
        state_bits=4,
    )
    initial = _build_bank_image(
        profile, mean=0.0, sigma=0.35, threshold=0.24, source_window=-1
    )
    update = _build_bank_image(
        profile, mean=0.55, sigma=0.45, threshold=0.24, source_window=0
    )
    rng = np.random.default_rng(9)
    bank = LUTParameterBank(initial)
    stale = bank.stage_and_commit(
        update,
        fault_mode="stale_commit",
        inject_fault=True,
        profile=profile,
        rng=rng,
    )
    assert stale["commit_applied"] is False
    assert bank.active_version == 0
    assert bank.active_image.image_sha256 == initial.image_sha256

    sign = bank.stage_and_commit(
        update,
        fault_mode="lut_sign_burst",
        inject_fault=True,
        profile=profile,
        rng=rng,
    )
    assert sign["commit_applied"] is True
    assert sign["address_count"] == 2
    assert bank.active_version == 1
    assert bank.active_image.image_sha256 != update.image_sha256
    assert bank.fault_counts == {"stale_commit": 1, "lut_sign_burst": 1}


def test_quantized_simulation_is_deterministic_integer_and_fault_detecting() -> None:
    cfg = FixedPointStressConfig(
        n_samples=4096,
        window_size=128,
        seeds=(13, 17),
        bootstrap_replicates=1000,
        bank_fault_every_updates=2,
    )
    trace = _physical_trace(cfg, 13)
    reference = _simulate_float_reference(trace, cfg)
    profile = base_profile()
    first = simulate_quantized_profile(
        profile,
        trace=trace,
        float_reference=reference,
        config=cfg,
        seed=13,
    )
    second = simulate_quantized_profile(
        profile,
        trace=trace,
        float_reference=reference,
        config=cfg,
        seed=13,
    )
    assert first == second
    assert first["integrity"]["version_matches_commits"] is True
    assert first["integrity"]["observed_only_estimator"] is True
    assert first["target_hardware_measured"] is False
    assert first["synthesis_measured"] is False

    faulty = simulate_quantized_profile(
        profile,
        trace=trace,
        float_reference=reference,
        config=cfg,
        seed=13,
        bank_fault_mode="state_msb_flip",
    )
    assert faulty["bank"]["events"]["state_msb_flip"] > 0
    assert faulty["bank"]["fault_counts"]["state_msb_flip"] > 0
    assert faulty["bank"]["active_version"] == faulty["bank"]["commits"]


def test_production_artifact_has_all_curves_faults_and_live_source_hash() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["contract_id"] == "T243-BIT-ACCURATE-PRECISION-RESOURCE-LER-V1"
    assert artifact["status"] == "PASS"
    assert all(artifact["gates"].values())
    assert artifact["target_hardware_measured"] is False
    assert artifact["synthesis_measured"] is False
    assert artifact["config"]["n_samples"] >= 65_536
    assert len(artifact["config"]["seeds"]) >= 8
    assert len(artifact["profiles"]) == 42
    assert len(artifact["per_seed_results"]) == (42 + 4) * 8
    assert set(artifact["bank_fault_aggregates"]) == set(BANK_FAULT_MODES[1:])
    for mode, row in artifact["bank_fault_aggregates"].items():
        assert "paired_ler_minus_base_quantized" in row
        if mode in {"lut_sign_burst", "state_msb_flip"}:
            assert row["paired_ler_minus_base_quantized"]["ci_low"] > artifact[
                "config"
            ]["severe_fault_min_ler_increase"]

    expected_lengths = {
        "adc_bits": 6,
        "lut_address_bits": 6,
        "llr_fractional_bits": 6,
        "threshold_fractional_bits": 6,
        "state_bits": 6,
        "update_period_windows": 6,
        "joint_precision": 5,
    }
    assert {key: len(value) for key, value in artifact["curves"].items()} == expected_lengths
    for row in artifact["per_seed_results"]:
        assert row["integrity"]["version_matches_commits"] is True
        assert row["resource_proxy"]["target_synthesis_measured"] is False
        assert row["resource_proxy"]["fpga_lut_count"] is None
        if row["bank_fault_mode"] != "none":
            assert row["bank"]["events"].get(row["bank_fault_mode"], 0) > 0

    high = artifact["curves"]["joint_precision"][-1]
    assert abs(high["paired_ler_minus_float"]["mean"]) <= artifact["config"][
        "high_precision_max_abs_delta_ler"
    ]
    assert artifact["pareto_candidates"]
    forbidden = " ".join(artifact["claim_boundary"]["forbidden"]).lower()
    assert "fpga lut/bram/dsp" in forbidden
    assert "post-place-and-route" in forbidden

    expected_hash = _sha256_sources(
        [
            ROOT / "cnn_fpga" / "runtime" / "fixed_point_chain.py",
            ROOT / "cnn_fpga" / "decoder" / "linear_runtime.py",
            ROOT / "physics" / "ideal_gkp_decoder.py",
        ]
    )
    assert artifact["implementation_sha256"] == expected_hash


def test_production_source_csv_is_complete_and_matches_json_rows() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    with SOURCE_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(artifact["per_seed_results"])
    assert {row["bank_fault_mode"] for row in rows} == set(BANK_FAULT_MODES)
    assert all(row["target_hardware_measured"] == "False" for row in rows)
    assert all(row["synthesis_measured"] == "False" for row in rows)


def test_figure_bundle_is_hash_bound_editable_and_python_only() -> None:
    audit = json.loads(DEFAULT_FIGURE_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert all(audit["gates"].values())
    assert audit["figure_contract"]["backend"] == "python_matplotlib_only"
    assert audit["figure_contract"]["final_width_mm"] == pytest.approx(182.88)
    assert audit["source_artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    for name in ("svg", "pdf", "tiff", "png"):
        output = ROOT / audit["outputs"][name]["path"]
        assert output.stat().st_size == audit["outputs"][name]["size_bytes"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == audit[
            "outputs"
        ][name]["sha256"]
    svg = (ROOT / audit["outputs"]["svg"]["path"]).read_text(encoding="utf-8")
    assert svg.count("<text") > 20
    assert "font:" in svg
