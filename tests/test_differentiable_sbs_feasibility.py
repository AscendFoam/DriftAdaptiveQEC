from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch is optional in the minimal recovery interpreter", allow_module_level=True)

from physics.differentiable_sbs_feasibility import (
    DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE,
    POLICY_ARCHITECTURE_ID,
    FeasibilityRecurrentPolicy,
    RecurrentPolicySpec,
    TrainingPointConfig,
    TrainingPointResult,
    benchmark_training_point,
    default_scan_points,
    run_point_subprocess,
    summarize_scan,
    validate_production_design,
)

import torch


ARTIFACT = Path("docs/t2_3_6_differentiable_sbs_feasibility.json")
SOURCE_CSV = Path("docs/t2_3_6_differentiable_sbs_feasibility.csv")
FIGURE_STEM = Path("docs/figures/t2_3_6_differentiable_sbs_feasibility")


def test_policy_spec_matches_registered_paper_scale_architecture() -> None:
    spec = RecurrentPolicySpec()
    assert spec.architecture_id == POLICY_ARCHITECTURE_ID
    assert spec.hidden_size == 10
    assert spec.dense_widths == (256, 256)
    assert spec.output_controls == 15
    assert spec.analytic_parameter_count == 72913


@pytest.mark.parametrize(
    "change,error",
    [
        ({"input_features": 2}, ValueError),
        ({"hidden_size": 0}, ValueError),
        ({"dense_widths": (256,)}, ValueError),
        ({"dense_widths": (256, 0)}, ValueError),
        ({"output_controls": 14}, ValueError),
        ({"output_scale": 0.0}, ValueError),
        ({"architecture_id": "demo"}, ValueError),
    ],
)
def test_policy_spec_negative_paths(change: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        RecurrentPolicySpec(**change)


def test_training_point_contract_and_memory_lower_bound_scale() -> None:
    point = TrainingPointConfig(cutoff=8, batch_size=4, full_cycles=2)
    assert point.half_cycles == 4
    assert point.point_id == "cpu-c8-b4-h2-float64"
    assert point.state_tensor_bytes == 4 * 16 * 16 * 16
    assert point.autograd_state_lower_bound_bytes == point.state_tensor_bytes * 29
    assert point.protocol_id == "PROTO-SBS-MAIN"
    assert "not optimizer convergence" in point.scope


@pytest.mark.parametrize(
    "change,error",
    [
        ({"cutoff": 3}, ValueError),
        ({"cutoff": 49}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"batch_size": 4097}, ValueError),
        ({"full_cycles": 0}, ValueError),
        ({"full_cycles": 11}, ValueError),
        ({"device": "tpu"}, ValueError),
        ({"real_dtype": "float16"}, ValueError),
        ({"grid_points": 1024}, ValueError),
        ({"warmup_steps": -1}, ValueError),
        ({"repeats": 0}, ValueError),
        ({"repeats": 21}, ValueError),
        ({"score_baseline": float("nan")}, ValueError),
        ({"runtime_budget_seconds": 0.0}, ValueError),
        ({"maximum_memory_fraction": 0.04}, ValueError),
        ({"seed": True}, TypeError),
        ({"policy": "demo"}, TypeError),
        ({"protocol_id": "wrong"}, ValueError),
        ({"scope": "demo"}, ValueError),
    ],
)
def test_training_point_negative_paths(change: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        TrainingPointConfig(**change)


def test_recurrent_policy_has_exact_count_shape_and_finite_gradients() -> None:
    policy = FeasibilityRecurrentPolicy(seed=7)
    assert policy.parameter_count == 72913
    history = torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.int64)
    output = policy(history, 3)
    assert output.shape == (2, 15)
    assert torch.all(torch.isfinite(output))
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in policy.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)


def test_recurrent_policy_is_prefix_causal_and_history_sensitive() -> None:
    policy = FeasibilityRecurrentPolicy(seed=11)
    prefix = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    same = policy(prefix, 2)
    assert torch.allclose(same[0], same[1])
    changed = policy(torch.tensor([[0, 1], [1, 1]], dtype=torch.int64), 2)
    assert not torch.allclose(changed[0], changed[1])
    with pytest.raises(ValueError, match="history width"):
        policy(prefix, 1)


@pytest.fixture(scope="module")
def cpu_point_result() -> TrainingPointResult:
    return benchmark_training_point(
        TrainingPointConfig(
            cutoff=4,
            batch_size=2,
            full_cycles=1,
            warmup_steps=0,
            repeats=2,
        )
    )


def test_real_cpu_point_executes_repeated_forward_reward_score_backward(
    cpu_point_result: TrainingPointResult,
) -> None:
    result = cpu_point_result
    assert result.status == "pass"
    assert result.feasible and result.preferred
    assert result.policy_parameter_count == 72913
    assert len(result.runtime_seconds) == 2
    assert result.minimum_gradient_norm is not None
    assert result.minimum_gradient_norm > 1.0e-10
    assert result.minimum_parameter_update_norm is not None
    assert result.minimum_parameter_update_norm > 0.0
    assert result.objective_finite and result.gradients_finite


def test_real_cpu_point_measures_memory_and_density_numerics(
    cpu_point_result: TrainingPointResult,
) -> None:
    result = cpu_point_result
    assert result.peak_rss_bytes is not None and result.peak_rss_bytes > 0
    assert result.rss_delta_bytes is not None and result.rss_delta_bytes > 0
    assert result.memory_amplification_over_lower_bound is not None
    assert result.maximum_trace_error is not None and result.maximum_trace_error < 1.0e-12
    assert result.maximum_hermiticity_error == 0.0
    assert result.minimum_final_eigenvalue is not None
    assert result.minimum_final_eigenvalue > -1.0e-10
    assert result.minimum_trajectory_probability is not None
    assert 0.0 < result.minimum_trajectory_probability <= 1.0


def test_subprocess_isolation_roundtrips_a_structured_point() -> None:
    result = run_point_subprocess(
        TrainingPointConfig(
            cutoff=4,
            batch_size=1,
            full_cycles=1,
            warmup_steps=0,
            repeats=1,
        ),
        timeout_seconds=120.0,
    )
    assert result.status == "pass"
    assert result.point_id == "cpu-c4-b1-h1-float64"
    assert result.failure_kind is None


def test_default_cuda_design_is_multiaxis_and_covers_every_horizon() -> None:
    points = default_scan_points("cuda")
    assert len(points) == 56
    validate_production_design(points, "cuda")
    assert {point.cutoff for point in points} >= {8, 12, 16, 18, 24, 32, 48}
    assert {point.batch_size for point in points} >= {1, 4, 8, 16, 32, 64, 128, 256, 512, 576}
    assert {point.full_cycles for point in points} >= set(range(2, 11))
    matrix = {
        (point.batch_size, point.full_cycles)
        for point in points
        if point.cutoff == 16
    }
    assert {(batch, horizon) for batch in (8, 16) for horizon in range(2, 11)} <= matrix
    assert {(576, 10), (512, 10), (256, 10), (128, 10)} <= matrix


def test_default_cpu_design_is_a_nontrivial_rss_fallback_lane() -> None:
    points = default_scan_points("cpu")
    assert len(points) == 9
    validate_production_design(points, "cpu")
    assert max(point.cutoff for point in points) == 16
    assert max(point.batch_size for point in points) == 8
    assert max(point.full_cycles for point in points) == 10


def test_production_design_rejects_demo_sized_or_mixed_device_lists() -> None:
    tiny = (TrainingPointConfig(),)
    with pytest.raises(ValueError, match="too small"):
        validate_production_design(tiny, "cpu")
    mixed = (TrainingPointConfig(device="cpu"), TrainingPointConfig(device="cuda"))
    with pytest.raises(ValueError, match="requested device"):
        validate_production_design(mixed, "cpu")


def _synthetic_result(
    *,
    cutoff: int,
    batch: int,
    horizon: int,
    feasible: bool = True,
) -> TrainingPointResult:
    point = TrainingPointConfig(
        cutoff=cutoff,
        batch_size=batch,
        full_cycles=horizon,
        device="cuda",
    )
    return TrainingPointResult(
        point_id=point.point_id,
        status="pass" if feasible else "runtime_exceeded",
        cutoff=cutoff,
        batch_size=batch,
        full_cycles=horizon,
        half_cycles=2 * horizon,
        device="cuda",
        real_dtype="float64",
        policy_architecture_id=POLICY_ARCHITECTURE_ID,
        policy_parameter_count=72913,
        warmup_steps=1,
        repeats=3,
        runtime_budget_seconds=10.0,
        preferred_runtime_seconds=2.0,
        maximum_memory_fraction=0.75,
        grid_points=2049,
        score_baseline=0.35,
        learning_rate=1.0e-4,
        seed=314159,
        numerical_stable=True,
        within_runtime_budget=feasible,
        within_memory_budget=True,
        feasible=feasible,
        preferred=feasible,
        objective_finite=True,
        gradients_finite=True,
    )


def test_scan_summary_requires_all_horizons_and_high_cutoff_anchor() -> None:
    results = [
        _synthetic_result(cutoff=16, batch=8, horizon=horizon)
        for horizon in range(2, 11)
    ]
    results.append(_synthetic_result(cutoff=18, batch=4, horizon=10))
    summary = summarize_scan(results)
    assert summary["two_to_ten_cycle_envelope_confirmed"]
    assert summary["common_feasible_batch_for_cycles_2_to_10"] == 8
    assert summary["decision"] == "FEASIBLE_2_TO_10_CYCLE_TEACHER_KERNEL"

    missing = summarize_scan(results[:-2])
    assert not missing["two_to_ten_cycle_envelope_confirmed"]
    assert missing["decision"] == "FALSIFIED_AT_REGISTERED_RESOURCE_GATE"


def test_scan_summary_tracks_larger_batch_without_overwriting_common_floor() -> None:
    results = [
        _synthetic_result(cutoff=16, batch=8, horizon=horizon)
        for horizon in range(2, 11)
    ]
    results.extend(
        _synthetic_result(cutoff=16, batch=16, horizon=horizon)
        for horizon in (2, 4, 6, 8, 10)
    )
    results.append(_synthetic_result(cutoff=24, batch=4, horizon=10))
    summary = summarize_scan(results)
    assert summary["largest_tested_feasible_batch_by_horizon_at_cutoff16"]["2"] == 16
    assert summary["largest_tested_feasible_batch_by_horizon_at_cutoff16"]["3"] == 8
    assert summary["common_feasible_batch_for_cycles_2_to_10"] == 8


def test_production_artifact_is_multiaxis_repeated_and_fail_closed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["task_id"] == "T2.3.6"
    assert payload["status"] == "PASS"
    assert payload["protocol_id"] == "PROTO-SBS-MAIN"
    assert payload["scope"] == DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE
    assert payload["policy_spec"]["analytic_parameter_count"] == 72913
    assert payload["measurement_contract"]["subprocess_isolation"] is True
    assert payload["measurement_contract"]["optimizer"] == "Adam"
    assert payload["measurement_contract"]["timed_repeats"] >= 3
    assert len(payload["points"]) == 65
    assert {point["device"] for point in payload["points"]} == {"cpu", "cuda"}
    assert set(payload["checks"].values()) == {True}
    assert payload["execution"]["reused_contract_identical_points"] + payload["execution"]["newly_executed_points"] == 65
    assert payload["host"]["torch"]
    assert payload["summary"]["decision"] in {
        "FEASIBLE_2_TO_10_CYCLE_TEACHER_KERNEL",
        "FALSIFIED_AT_REGISTERED_RESOURCE_GATE",
    }
    assert "not optimizer convergence" in payload["scope"]


def test_publication_figure_bundle_is_editable_and_source_traceable() -> None:
    rows = SOURCE_CSV.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 66
    assert "runtime_repeat_3_seconds" in rows[0]
    for suffix in ("svg", "pdf", "tiff", "png"):
        path = FIGURE_STEM.with_suffix(f".{suffix}")
        assert path.is_file() and path.stat().st_size > 1000
    svg = FIGURE_STEM.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "not sampled" in svg
    assert "75% VRAM gate" in svg
