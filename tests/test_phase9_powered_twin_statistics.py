from __future__ import annotations

import copy

import numpy as np
import pytest

from cnn_fpga.benchmark.phase9_powered_twin_statistics import (
    ObservedGateInput,
    ObservedConservativeGateInput,
    evaluate_gate,
    evaluate_conservative_gate,
    exact_mapping_gate_results,
    joint_maxt,
    joint_maxt_stream,
    rademacher_matrix,
)


def _roots(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}/cluster={index:04d}" for index in range(count))


def _scalar_gate(
    gate_id: str,
    values: np.ndarray,
    *,
    roots: tuple[str, ...] | None = None,
    margin: float = 1.0,
    quantization: float = 0.0,
) -> ObservedGateInput:
    return ObservedGateInput(
        gate_id=gate_id,
        family="scalar",
        direction="upper",
        margin=margin,
        kind="scalar_abs_mean",
        cluster_root_ids=roots or _roots(gate_id, len(values)),
        values=np.asarray(values, dtype=np.float64),
        quantization_radius=quantization,
    )


def test_common_root_reuses_sign_but_independent_root_does_not_alias() -> None:
    first_roots = _roots("scope-a", 32)
    same = rademacher_matrix(
        seed=100_050_000_000,
        replicates=199,
        cluster_root_ids=first_roots,
    )
    replay = rademacher_matrix(
        seed=100_050_000_000,
        replicates=199,
        cluster_root_ids=first_roots,
    )
    other = rademacher_matrix(
        seed=100_050_000_000,
        replicates=199,
        cluster_root_ids=_roots("scope-b", 32),
    )
    assert np.array_equal(same, replay)
    assert not np.array_equal(same, other)
    assert set(np.unique(same)) == {-1, 1}


def test_scalar_multiplier_uses_functional_of_mean_not_mean_of_norms() -> None:
    values = np.asarray([-4.0, -2.0, 1.0, 5.0])
    gate = _scalar_gate("g", values)
    signs = rademacher_matrix(
        seed=17,
        replicates=199,
        cluster_root_ids=gate.cluster_root_ids,
    )
    result = evaluate_gate(gate, signs=signs)
    assert result.point == pytest.approx(abs(float(np.mean(values))))
    assert result.point != pytest.approx(float(np.mean(np.abs(values))))
    assert result.standard_error > 0.0
    assert result.replicate_values_sha256


def test_vector_and_density_are_bootstrapped_in_native_environment() -> None:
    rng = np.random.default_rng(11)
    vector_values = rng.normal(size=(24, 3))
    vector = ObservedGateInput(
        gate_id="vector",
        family="vector",
        direction="upper",
        margin=2.0,
        kind="vector_l2_mean",
        cluster_root_ids=_roots("vector", 24),
        values=vector_values,
        quantization_radius=0.01,
    )
    vector_result = evaluate_gate(
        vector,
        signs=rademacher_matrix(
            seed=18,
            replicates=199,
            cluster_root_ids=vector.cluster_root_ids,
        ),
    )
    assert vector_result.point == pytest.approx(
        np.linalg.norm(np.mean(vector_values, axis=0))
    )

    raw = rng.normal(size=(24, 4, 4)) + 1j * rng.normal(size=(24, 4, 4))
    density_values = raw + np.swapaxes(raw.conj(), 1, 2)
    density_values -= np.mean(density_values, axis=0, keepdims=True)
    # Add a small nonzero point while preserving Hermiticity.
    density_values[:, 0, 0] += 0.02
    density = ObservedGateInput(
        gate_id="density",
        family="density",
        direction="upper",
        margin=1.0,
        kind="density_half_trace_mean",
        cluster_root_ids=_roots("density", 24),
        values=density_values,
        quantization_radius=0.001,
    )
    density_result = evaluate_gate(
        density,
        signs=rademacher_matrix(
            seed=19,
            replicates=199,
            cluster_root_ids=density.cluster_root_ids,
        ),
    )
    expected = 0.5 * np.sum(
        np.abs(np.linalg.eigvalsh(np.mean(density_values, axis=0)))
    )
    assert density_result.point == pytest.approx(expected)
    assert density_result.standard_error > 0.0


def test_vector_l1_is_not_silently_replaced_by_l2() -> None:
    values = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 3.5],
        ]
    )
    gate = ObservedGateInput(
        gate_id="l1",
        family="level_l1",
        direction="upper",
        margin=10.0,
        kind="vector_l1_mean",
        cluster_root_ids=_roots("l1", len(values)),
        values=values,
        quantization_radius=0.0,
    )
    result = evaluate_gate(
        gate,
        signs=rademacher_matrix(
            seed=99,
            replicates=199,
            cluster_root_ids=gate.cluster_root_ids,
        ),
    )
    mean = np.mean(values, axis=0)
    assert result.point == pytest.approx(np.linalg.norm(mean, ord=1))
    assert result.point != pytest.approx(np.linalg.norm(mean, ord=2))


def test_conservative_cutoff_gate_keeps_two_leg_replicates_and_bounds() -> None:
    roots = _roots("cutoff", 32)
    gate = ObservedConservativeGateInput(
        gate_id="cutoff/36-40/shared/vacuum_g/IDLE/density",
        family="cutoff_mapping",
        direction="upper",
        margin=5.0,
        kind="scalar_abs_mean",
        cluster_root_ids=roots,
        leg_a_values=np.linspace(-0.5, 0.8, 32),
        leg_b_values=np.sin(np.linspace(0.0, 5.0, 32)),
        leg_a_quantization_radius=0.01,
        leg_b_quantization_radius=0.03,
    )
    signs = rademacher_matrix(
        seed=101,
        replicates=199,
        cluster_root_ids=roots,
    )
    result = evaluate_conservative_gate(gate, signs=signs)
    assert len(result.bound_components) == 2
    assert len(result.leg_details) == 2
    assert np.array_equal(
        result.standardized_replicate_errors,
        np.maximum(
            evaluate_gate(
                _scalar_gate(
                    "cutoff/36-40/shared/vacuum_g/IDLE/density",
                    gate.leg_a_values,
                    roots=roots,
                    margin=5.0,
                    quantization=0.01,
                ),
                signs=signs,
            ).standardized_replicate_errors,
            evaluate_gate(
                _scalar_gate(
                    "cutoff/36-40/shared/vacuum_g/IDLE/density",
                    gate.leg_b_values,
                    roots=roots,
                    margin=5.0,
                    quantization=0.03,
                ),
                signs=signs,
            ).standardized_replicate_errors,
        ),
    )
    report = joint_maxt_stream(
        iter((gate,)),
        expected_gate_ids=(gate.gate_id,),
        rademacher_seed=101,
        expected_stochastic_gate_count=1,
    )
    row = report["gates"][0]
    expected_bound = max(
        item["point"]
        + report["joint_critical"] * item["standard_error"]
        + item["quantization_radius"]
        for item in row["leg_details"]
    )
    assert row["simultaneous_bound"] == pytest.approx(expected_bound)


def test_degenerate_observed_se_is_incomplete_not_floor_rescued() -> None:
    gate = _scalar_gate("degenerate", np.ones(16))
    with pytest.raises(RuntimeError, match="INCOMPLETE_FAIL_CLOSED"):
        evaluate_gate(
            gate,
            signs=rademacher_matrix(
                seed=20,
                replicates=199,
                cluster_root_ids=gate.cluster_root_ids,
            ),
        )


def test_closed_family_joint_critical_and_external_quantization() -> None:
    roots = _roots("shared", 48)
    first = _scalar_gate(
        "g1",
        np.linspace(-0.4, 0.6, 48),
        roots=roots,
        margin=2.0,
    )
    second = _scalar_gate(
        "g2",
        np.sin(np.linspace(0.0, 4.0, 48)),
        roots=roots,
        margin=2.0,
        quantization=0.05,
    )
    report = joint_maxt(
        [second, first],
        rademacher_seed=21,
        expected_stochastic_gate_count=2,
    )
    assert report["stochastic_gate_count"] == 2
    assert report["multiplier_replicates"] == 199
    assert len(report["maxima"]) == 199
    assert report["quantile_method"] == "higher"
    sorted_maxima = sorted(report["maxima"])
    assert report["joint_critical"] == sorted_maxima[189]
    by_id = {row["gate_id"]: row for row in report["gates"]}
    assert by_id["g2"]["simultaneous_bound"] == pytest.approx(
        by_id["g2"]["point"]
        + report["joint_critical"] * by_id["g2"]["standard_error"]
        + 0.05
    )
    assert report["synthetic_influence_used"] is False
    assert report["pointwise_z_substitution_used"] is False
    assert report["family_split_used"] is False


def test_gate_deletion_duplicate_and_wrong_B_fail_closed() -> None:
    gate = _scalar_gate("g1", np.linspace(-1.0, 1.0, 16))
    with pytest.raises(RuntimeError, match="count drift"):
        joint_maxt(
            [gate],
            rademacher_seed=22,
            expected_stochastic_gate_count=2,
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        joint_maxt(
            [gate, copy.deepcopy(gate)],
            rademacher_seed=22,
            expected_stochastic_gate_count=2,
        )
    with pytest.raises(ValueError, match="B=199"):
        rademacher_matrix(
            seed=22,
            replicates=198,
            cluster_root_ids=gate.cluster_root_ids,
        )


def test_one_raw_mutation_changes_gate_and_global_replicate_fingerprints() -> None:
    first = _scalar_gate("g1", np.linspace(-1.0, 1.0, 32), margin=3.0)
    second = _scalar_gate("g2", np.linspace(0.1, 1.1, 32), margin=3.0)
    original = joint_maxt(
        [first, second],
        rademacher_seed=23,
        expected_stochastic_gate_count=2,
    )
    changed_values = np.array(first.values, copy=True)
    changed_values[7] += 0.125
    changed_first = _scalar_gate("g1", changed_values, margin=3.0)
    mutated = joint_maxt(
        [changed_first, second],
        rademacher_seed=23,
        expected_stochastic_gate_count=2,
    )
    original_by_id = {row["gate_id"]: row for row in original["gates"]}
    mutated_by_id = {row["gate_id"]: row for row in mutated["gates"]}
    assert (
        original_by_id["g1"]["replicate_values_sha256"]
        != mutated_by_id["g1"]["replicate_values_sha256"]
    )
    assert (
        original["replicate_fingerprint_sha256"]
        != mutated["replicate_fingerprint_sha256"]
    )
    assert original["maxima_sha256"] != mutated["maxima_sha256"]


def test_streaming_closed_family_matches_in_memory_and_rejects_substitution() -> None:
    roots = _roots("stream", 24)
    gates = [
        _scalar_gate(
            "g1",
            np.linspace(-0.2, 0.9, 24),
            roots=roots,
            margin=4.0,
        ),
        _scalar_gate(
            "g2",
            np.cos(np.linspace(0.0, 2.0, 24)),
            roots=roots,
            margin=4.0,
        ),
    ]
    memory = joint_maxt(
        gates,
        rademacher_seed=123,
        expected_stochastic_gate_count=2,
    )
    streamed = joint_maxt_stream(
        (gate for gate in reversed(gates)),
        expected_gate_ids=("g1", "g2"),
        rademacher_seed=123,
        expected_stochastic_gate_count=2,
    )
    assert streamed["analysis_sha256"] == memory["analysis_sha256"]
    with pytest.raises(RuntimeError, match="coverage drift"):
        joint_maxt_stream(
            iter(gates[:1]),
            expected_gate_ids=("g1", "g2"),
            rademacher_seed=123,
            expected_stochastic_gate_count=2,
        )


def test_exact_mapping_gate_set_is_six_and_all_required() -> None:
    rows = []
    for cutoff in (36, 40, 44):
        rows.extend(
            [
                {
                    "gate_id": f"mapping/c{cutoff}/principal_singular",
                    "point": 0.97,
                },
                {
                    "gate_id": f"mapping/c{cutoff}/projector_frobenius",
                    "point": 0.2,
                },
            ]
        )
    results = exact_mapping_gate_results(rows)
    assert len(results) == 6
    assert all(row["passed"] for row in results)
    with pytest.raises(RuntimeError, match="exactly six"):
        exact_mapping_gate_results(rows[:-1])
