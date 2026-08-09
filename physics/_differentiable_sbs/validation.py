"""Private validation and CLI for the differentiable SBS simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..differentiable_sbs_trajectory import (
    DIFFERENTIABLE_SBS_SCOPE,
    PARAMETER_NAMES,
    POROTTI_S1_SOURCE,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
    TorchDeviceName,
    _require_torch,
)


def run_differentiable_sbs_validation(
    *,
    device: TorchDeviceName = "cpu",
    cutoff: int = 8,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic production gates and optionally write their JSON artifact."""

    th = _require_torch()
    config = DifferentiableSBSConfig(
        cutoff=cutoff,
        full_cycles=1,
        batch_size=4,
        grid_points=4097,
        device=device,
        real_dtype="float64",
    )
    simulator = DifferentiableSBSTrajectorySimulator(config)
    raw = th.linspace(
        -0.08,
        0.08,
        steps=config.half_cycles * len(PARAMETER_NAMES),
        dtype=simulator.real_dtype,
        device=simulator.device,
    ).reshape(1, config.half_cycles, len(PARAMETER_NAMES))
    raw = raw.expand(config.batch_size, -1, -1).clone().requires_grad_(True)
    all_branches = th.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=th.int64,
        device=simulator.device,
    )
    result = simulator.run(raw, forced_outcomes=all_branches, seed=314159)
    branch_probability_sum = float(result.trajectory_probability.sum().detach().cpu())
    scalar = result.reward.mean() + 0.01 * result.log_probability.mean()
    gradient = th.autograd.grad(scalar, raw, retain_graph=False)[0]
    gradient_norm = float(th.linalg.vector_norm(gradient).detach().cpu())
    policy_bias = th.zeros(
        (len(PARAMETER_NAMES),),
        dtype=simulator.real_dtype,
        device=simulator.device,
        requires_grad=True,
    )
    history_direction = th.linspace(
        -0.12,
        0.12,
        len(PARAMETER_NAMES),
        dtype=simulator.real_dtype,
        device=simulator.device,
    )

    def history_policy(history: Any, half_index: int) -> Any:
        if half_index == 0:
            signed_latest = th.zeros(
                (config.batch_size, 1),
                dtype=simulator.real_dtype,
                device=simulator.device,
            )
        else:
            signed_latest = 2.0 * history[:, -1:].to(simulator.real_dtype) - 1.0
        return policy_bias[None, :] + signed_latest * history_direction[None, :]

    policy_result = simulator.run(
        control_policy=history_policy,
        forced_outcomes=all_branches,
        seed=314159,
    )
    policy_scalar = policy_result.reward.mean() + 0.01 * policy_result.log_probability.mean()
    policy_gradient = th.autograd.grad(policy_scalar, policy_bias)[0]
    policy_gradient_norm = float(th.linalg.vector_norm(policy_gradient).detach().cpu())
    history_control_separation = float(
        th.linalg.vector_norm(
            policy_result.physical_controls[0, 1]
            - policy_result.physical_controls[2, 1]
        )
        .detach()
        .cpu()
    )
    replay = simulator.run(raw.detach(), forced_outcomes=all_branches, seed=9)
    gate_residuals = simulator.gate_unitarity_residuals(raw.detach())
    idle_residuals = simulator.idle_completeness_residuals()
    max_gate_residual = max(gate_residuals.values())
    max_idle_residual = max(idle_residuals.values())
    checks = {
        "fixed_half_cycle_is_5000_ns": config.timing.half_cycle_duration_ns == 5000,
        "fixed_full_cycle_is_10000_ns": config.timing.full_cycle_duration_ns == 10000,
        "all_four_two_measurement_branches_normalize": abs(branch_probability_sum - 1.0)
        < 2.0e-9,
        "trajectory_probability_matches_log_probability": bool(
            th.allclose(
                result.trajectory_probability,
                th.exp(result.log_probability),
                atol=2.0e-12,
                rtol=2.0e-12,
            )
        ),
        "forced_replay_is_exact": bool(
            th.allclose(result.reward.detach(), replay.reward, atol=2.0e-12, rtol=2.0e-12)
            and th.allclose(
                result.trajectory_probability.detach(),
                replay.trajectory_probability,
                atol=2.0e-12,
                rtol=2.0e-12,
            )
        ),
        "reward_is_physical": bool(
            th.all((result.reward.detach() >= 0.0) & (result.reward.detach() <= 1.0))
        ),
        "autograd_graph_is_connected": bool(
            th.all(th.isfinite(gradient)).detach().cpu() and gradient_norm > 1.0e-10
        ),
        "history_conditioned_policy_changes_future_controls": history_control_separation
        > 1.0e-8,
        "history_conditioned_policy_gradient_is_connected": bool(
            th.all(th.isfinite(policy_gradient)).detach().cpu()
            and policy_gradient_norm > 1.0e-10
        ),
        "history_conditioned_branch_tree_normalizes": abs(
            float(policy_result.trajectory_probability.sum().detach().cpu()) - 1.0
        )
        < 2.0e-9,
        "gate_families_are_unitary": max_gate_residual < 2.0e-10,
        "idle_channels_are_trace_preserving": max_idle_residual < 2.0e-10,
        "trajectory_trace_is_preserved": result.maximum_trace_error < 2.0e-10,
        "trajectory_is_hermitian": result.maximum_hermiticity_error < 2.0e-10,
        "final_density_is_positive": result.minimum_final_eigenvalue > -2.0e-10,
        "resource_profile_counts_fifteen_controls_per_half_cycle": (
            result.resource_profile.trainable_controls
            == config.batch_size * config.half_cycles * 15
        ),
        "resource_profile_is_not_hardware_measurement": not result.resource_profile.target_hardware_measured,
    }
    payload = {
        "task_id": "T2.3.4",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_contract": {
            "primary_source": POROTTI_S1_SOURCE,
            "gate_definitions": (
                "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction.md:411-434"
            ),
            "feedback_grape_probability_gradient": (
                "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction.md:467-495"
            ),
            "parameter_contract": (
                "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction.md:511-527"
            ),
        },
        "config": {**asdict(config), "timing": asdict(config.timing)},
        "metrics": {
            "branch_probability_sum": branch_probability_sum,
            "gradient_norm": gradient_norm,
            "policy_gradient_norm": policy_gradient_norm,
            "history_control_separation": history_control_separation,
            "maximum_gate_unitarity_residual": max_gate_residual,
            "maximum_idle_completeness_residual": max_idle_residual,
            "maximum_trace_error": result.maximum_trace_error,
            "maximum_hermiticity_error": result.maximum_hermiticity_error,
            "minimum_final_eigenvalue": result.minimum_final_eigenvalue,
            "reward_minimum": float(th.min(result.reward).detach().cpu()),
            "reward_maximum": float(th.max(result.reward).detach().cpu()),
            "trajectory_probability_minimum": float(
                th.min(result.trajectory_probability).detach().cpu()
            ),
            "trajectory_probability_maximum": float(
                th.max(result.trajectory_probability).detach().cpu()
            ),
        },
        "resource_profile": asdict(result.resource_profile),
        "gate_unitarity_residuals": gate_residuals,
        "idle_completeness_residuals": idle_residuals,
        "checks": checks,
        "scope": DIFFERENTIABLE_SBS_SCOPE,
        "forbidden_claims": (
            "not a pulse-level cavity-transmon Hamiltonian simulation",
            "not a leakage/SPAM/device-calibrated model",
            "not a Feedback-GRAPE gradient validation (reserved for T2.3.5)",
            "not a cutoff/batch/horizon feasibility envelope (reserved for T2.3.6)",
            "not a standard/MF/NMF ranking (reserved for T2.3.7)",
            "not target-board latency or hardware evidence",
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument(
        "--output", default="docs/t2_3_4_differentiable_trajectory_validation.json"
    )
    args = parser.parse_args()
    payload = run_differentiable_sbs_validation(
        device=args.device, cutoff=args.cutoff, output=args.output
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_differentiable_sbs_validation"]
