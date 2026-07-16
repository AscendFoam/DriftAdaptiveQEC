from __future__ import annotations

import csv
import hashlib
import json
from math import log, pi
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.standard_binning_baseline import major_comparison_registry
from cnn_fpga.benchmark.topk_lattice_coset_map import (
    FULL_PERIODIC_GAUSSIAN_MAP_ID,
    TOPK_DECODER_DESCRIPTOR,
    TOPK_LATTICE_COSET_MAP_ID,
    topk_cost_profile,
    topk_map_decode_2d,
    topk_map_sweep_2d,
    validate_topk_comparison_registration,
)
from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import covariance_from_sigmas, map_decode_2d


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_1_5_topk_map_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_1_5_topk_map_source_data.csv"


def _covariance(rho: float = 0.72) -> np.ndarray:
    return covariance_from_sigmas(
        0.52 * LATTICE_CONST,
        0.37 * LATTICE_CONST,
        rho,
    )


def _independent_topk_log_likelihood(
    syndrome: np.ndarray,
    covariance: np.ndarray,
    mean: np.ndarray,
    parity: tuple[int, int],
    k: int,
    *,
    tail_sigma: float,
) -> float:
    inverse = np.linalg.inv(covariance)
    log_determinant = float(np.linalg.slogdet(covariance)[1])
    radius_q = max(
        2,
        int(np.ceil(tail_sigma * np.sqrt(covariance[0, 0]) / LATTICE_CONST)) + 2,
    )
    radius_p = max(
        2,
        int(np.ceil(tail_sigma * np.sqrt(covariance[1, 1]) / LATTICE_CONST)) + 2,
    )
    nearest = np.floor((mean - syndrome) / LATTICE_CONST + 0.5).astype(int)
    terms: list[float] = []
    for q_alias in range(nearest[0] - radius_q, nearest[0] + radius_q + 1):
        for p_alias in range(nearest[1] - radius_p, nearest[1] + radius_p + 1):
            if (q_alias % 2, p_alias % 2) != parity:
                continue
            residual = syndrome + LATTICE_CONST * np.array([q_alias, p_alias]) - mean
            terms.append(
                -0.5 * float(residual @ inverse @ residual)
                - log(2.0 * pi)
                - 0.5 * log_determinant
            )
    selected = np.sort(np.asarray(terms))[::-1][:k]
    maximum = float(selected[0])
    return maximum + float(np.log(np.sum(np.exp(selected - maximum))))


def test_descriptor_is_single_mode_and_explicitly_not_kmwm() -> None:
    descriptor = TOPK_DECODER_DESCRIPTOR

    assert descriptor.baseline_id == TOPK_LATTICE_COSET_MAP_ID
    assert descriptor.deployable
    assert descriptor.hidden_truth_inputs == ()
    assert "single_mode" in descriptor.evidence_scope
    assert "K_minimum_weight_matchings" in descriptor.excluded_algorithms
    assert "joint 2D Gaussian lattice-pair" in descriptor.approximation_rule


@pytest.mark.parametrize("k", [1, 2, 5])
def test_each_coset_topk_sum_matches_independent_alias_enumeration(k: int) -> None:
    covariance = _covariance()
    syndrome = np.array([0.31, -0.44])
    mean = np.array([0.27, -0.19])
    result = topk_map_decode_2d(
        syndrome,
        covariance,
        k,
        mean=mean,
        tail_sigma=8.0,
    )

    expected = np.empty((2, 2))
    for q_parity in (0, 1):
        for p_parity in (0, 1):
            expected[q_parity, p_parity] = _independent_topk_log_likelihood(
                syndrome,
                covariance,
                mean,
                (q_parity, p_parity),
                k,
                tail_sigma=8.0,
            )
    np.testing.assert_allclose(result.log_likelihoods, expected, atol=2.0e-15)


def test_saturated_k_matches_full_periodic_map_for_correlated_batch() -> None:
    covariance = _covariance(-0.88)
    rng = np.random.default_rng(315)
    syndrome = rng.uniform(
        -0.5 * LATTICE_CONST,
        np.nextafter(0.5 * LATTICE_CONST, -np.inf),
        size=(700, 2),
    )
    mean = np.array([0.41, -0.36])
    full = map_decode_2d(
        syndrome, covariance, mean=mean, tail_sigma=8.0
    )
    topk = topk_map_decode_2d(
        syndrome, covariance, 256, mean=mean, tail_sigma=8.0
    )

    assert topk.saturated_full_candidate_sum
    np.testing.assert_array_equal(topk.logical_class, full.logical_class)
    np.testing.assert_allclose(topk.log_likelihoods, full.log_likelihoods, atol=4.0e-15)
    np.testing.assert_allclose(topk.posterior, full.posterior, atol=8.0e-16)


def test_likelihood_is_monotone_in_k_and_never_exceeds_full_sum() -> None:
    covariance = _covariance()
    syndrome = np.array([[0.10, -0.20], [-0.70, 0.62], [0.88, -0.91]])
    sweep = topk_map_sweep_2d(
        syndrome,
        covariance,
        (1, 2, 4, 8, 16, 32, 64, 128),
        mean=(0.22, -0.17),
        tail_sigma=8.0,
    )
    full = map_decode_2d(
        syndrome, covariance, mean=(0.22, -0.17), tail_sigma=8.0
    )
    previous = None
    for result in sweep.values():
        if previous is not None:
            assert np.all(result.log_likelihoods >= previous - 2.0e-15)
        assert np.all(result.log_likelihoods <= full.log_likelihoods + 3.0e-15)
        previous = result.log_likelihoods


def test_sweep_and_individual_api_are_bitwise_consistent() -> None:
    covariance = _covariance(0.31)
    syndrome = np.array([[0.2, 0.3], [-0.4, 0.5]])
    sweep = topk_map_sweep_2d(
        syndrome, covariance, (1, 3, 9), mean=(0.1, -0.2), tail_sigma=7.0
    )
    for k, swept in sweep.items():
        direct = topk_map_decode_2d(
            syndrome, covariance, k, mean=(0.1, -0.2), tail_sigma=7.0
        )
        np.testing.assert_array_equal(swept.logical_class, direct.logical_class)
        np.testing.assert_array_equal(swept.log_likelihoods, direct.log_likelihoods)
        np.testing.assert_array_equal(swept.posterior, direct.posterior)


def test_even_lattice_translation_of_mean_preserves_logical_posterior() -> None:
    covariance = _covariance()
    syndrome = np.array([[0.12, -0.28], [-0.83, 0.74]])
    base = topk_map_decode_2d(
        syndrome, covariance, 8, mean=(0.23, -0.31), tail_sigma=8.0
    )
    shifted = topk_map_decode_2d(
        syndrome,
        covariance,
        8,
        mean=(0.23 + 2 * LATTICE_CONST, -0.31 - 2 * LATTICE_CONST),
        tail_sigma=8.0,
    )

    np.testing.assert_allclose(base.log_likelihoods, shifted.log_likelihoods, atol=3.0e-15)
    np.testing.assert_allclose(base.posterior, shifted.posterior, atol=8.0e-16)
    np.testing.assert_array_equal(base.logical_class, shifted.logical_class)


def test_scalar_result_has_consistent_class_parity_llr_and_tie_order() -> None:
    covariance = covariance_from_sigmas(0.3, 0.3, 0.0)
    result = topk_map_decode_2d((0.0, 0.0), covariance, 64)

    assert isinstance(result.logical_class, int)
    assert result.logical_class == 0
    assert result.parity.shape == (2,)
    assert result.q_llr >= 0.0
    assert result.p_llr >= 0.0
    assert np.sum(result.posterior) == pytest.approx(1.0, abs=1.0e-15)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"k": 0}, ValueError, "positive"),
        ({"k": True}, TypeError, "integer"),
        ({"k": 1, "syndrome": (0.0, 0.0, 0.0)}, ValueError, "shape"),
        ({"k": 1, "syndrome": (LATTICE_CONST / 2.0, 0.0)}, ValueError, "half-open"),
        ({"k": 1, "prior": ((0.5, 0.5), (0.0, 0.0))}, ValueError, "strictly positive"),
        ({"k": 1, "covariance": ((1.0, 2.0), (2.0, 1.0))}, ValueError, "positive definite"),
    ],
)
def test_invalid_decoder_inputs_fail_closed(kwargs, exception, message) -> None:
    values = {
        "syndrome": (0.0, 0.0),
        "covariance": _covariance(),
        "k": 1,
    }
    values.update(kwargs)
    with pytest.raises(exception, match=message):
        topk_map_decode_2d(**values)


def test_unbounded_alias_workload_fails_before_allocation() -> None:
    huge = covariance_from_sigmas(
        1000.0 * LATTICE_CONST,
        1000.0 * LATTICE_CONST,
        0.0,
    )
    with pytest.raises(ValueError, match="workload"):
        topk_map_decode_2d((0.0, 0.0), huge, 1, tail_sigma=10.0)


def test_cost_profile_is_deterministic_monotone_and_not_fake_synthesis() -> None:
    covariance = _covariance()
    profiles = [
        topk_cost_profile(
            covariance, k, mean=(0.2, -0.1), tail_sigma=8.0, value_bits=24
        )
        for k in (1, 2, 4, 8, 16, 32, 64, 128)
    ]

    assert all(
        first.retained_state_bits <= second.retained_state_bits
        and first.streaming_topk_comparisons_upper
        <= second.streaming_topk_comparisons_upper
        for first, second in zip(profiles, profiles[1:])
    )
    final = profiles[-1]
    assert final.saturated_full_candidate_sum
    assert final.gaussian_multiplications == 6 * final.candidate_aliases
    assert final.gaussian_additions == 4 * final.candidate_aliases
    assert final.target_lut is final.target_bram is final.target_dsp is None
    assert final.target_fmax_hz is None
    assert not final.target_measured
    assert "not_synthesis" in final.scope


def test_comparison_registration_is_same_decoder_sensitivity_not_ranking() -> None:
    gates = validate_topk_comparison_registration()
    entry = next(
        item
        for item in major_comparison_registry()
        if item.comparison_id == "t3_1_5_topk_periodic_map_sensitivity"
    )

    assert entry.method_ids == (
        FULL_PERIODIC_GAUSSIAN_MAP_ID,
        TOPK_LATTICE_COSET_MAP_ID,
    )
    assert entry.comparison_kind == "implementation_sensitivity"
    assert entry.standard_binning_policy == "not_applicable"
    assert gates[-1] == "registry:t3_1_5_topk_sensitivity"


def _implementation_hash() -> str:
    paths = (
        ROOT / "cnn_fpga" / "benchmark" / "topk_lattice_coset_map.py",
        ROOT / "physics" / "ideal_gkp_decoder.py",
        ROOT / "cnn_fpga" / "benchmark" / "standard_binning_baseline.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_production_artifact_is_source_bound_and_scans_to_exact_saturation() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_hash()
    assert payload["aggregate"]["samples"] == 288_000
    assert payload["aggregate"]["source_data_rows"] == 192
    assert payload["aggregate"]["K1_nonexact_scenarios"] == 6
    assert payload["aggregate"]["largest_K_exact_scenarios"] == 6
    assert payload["aggregate"]["convergence_K_min"] >= 2
    assert payload["aggregate"]["convergence_K_max"] <= 8
    assert payload["gate_summary"]["failed"] == 0
    for scenario in payload["scenarios"]:
        assert scenario["unique_trace_hashes"] == 4
        assert scenario["sweep"][0]["axis_llr_p99_abs_error"] > 1.0e-3
        assert scenario["sweep"][-1]["decision_disagreement_rate"] == 0.0
        assert scenario["sweep"][-1]["axis_llr_max_abs_error"] <= 5.0e-13


def test_source_data_recomputes_scenario_k_ler_and_has_unique_traces() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6 * 4 * 8
    assert len({(row["scenario_id"], row["evaluation_seed"], row["K"]) for row in rows}) == 192
    assert len({(row["scenario_id"], row["trace_sha256"]) for row in rows}) == 24
    for scenario in payload["scenarios"]:
        scenario_rows = [row for row in rows if row["scenario_id"] == scenario["scenario"]["scenario_id"]]
        for sweep_row in scenario["sweep"]:
            selected = [row for row in scenario_rows if int(row["K"]) == sweep_row["K"]]
            assert len(selected) == 4
            assert np.mean([float(row["full_map_ler"]) for row in selected]) == pytest.approx(
                sweep_row["full_map_ler"], abs=1.0e-15
            )
            assert np.mean([float(row["topk_map_ler"]) for row in selected]) == pytest.approx(
                sweep_row["topk_map_ler"], abs=1.0e-15
            )


def test_claim_boundary_rejects_surface_hardware_and_universal_k_claims() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    forbidden = payload["claim_boundary"]["forbidden"]

    assert "surface-code K-MWM" in forbidden
    assert "hardware synthesis" in forbidden
    assert "universal optimal K" in forbidden
    assert payload["comparison_contract"]["same_candidate_rectangle"] is True
