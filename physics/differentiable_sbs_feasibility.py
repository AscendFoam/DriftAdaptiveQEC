"""Production scan orchestration for differentiable SBS feasibility."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ._differentiable_sbs.worker import (
    DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE,
    POLICY_ARCHITECTURE_ID,
    RESULT_MARKER,
    FeasibilityRecurrentPolicy,
    RecurrentPolicySpec,
    TrainingPointConfig,
    TrainingPointResult,
    _require_runtime,
    benchmark_training_point,
    run_point_subprocess,
    safe_benchmark_training_point,
)
from .sbs_error_space import SBS_PROTOCOL_ID


def default_scan_points(
    device: str,
    *,
    warmup_steps: int = 1,
    repeats: int = 3,
    runtime_budget_seconds: float = 10.0,
    preferred_runtime_seconds: float = 2.0,
) -> tuple[TrainingPointConfig, ...]:
    """Return the preregistered staged scan; it is not a tiny Cartesian demo."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    common = dict(
        device=device,
        warmup_steps=warmup_steps,
        repeats=repeats,
        runtime_budget_seconds=runtime_budget_seconds,
        preferred_runtime_seconds=preferred_runtime_seconds,
    )
    triples: set[tuple[int, int, int]] = set()
    if device == "cuda":
        # Cutoff and batch axes at a two-cycle anchor.
        triples.update((cutoff, 8, 2) for cutoff in (8, 12, 16, 18, 24, 32, 48))
        triples.update((16, batch, 2) for batch in (1, 4, 8, 16, 32, 64, 128, 256, 512))
        # Every 2--10 cycle horizon at two useful training batches.
        triples.update((16, batch, horizon) for batch in (8, 16) for horizon in range(2, 11))
        # Expansion points test whether even horizons can use a larger batch.
        triples.update((16, 32, horizon) for horizon in (2, 4, 6, 8, 10))
        # High-cutoff long-horizon anchors prevent a cutoff-8-only conclusion.
        triples.update((cutoff, 4, horizon) for cutoff in (18, 24) for horizon in (6, 10))
        # Isolated resource-frontier probes continue until memory/runtime rejection.
        triples.update((16, batch, 10) for batch in (64, 128, 256, 512))
        triples.add((16, 576, 10))
        triples.update((24, batch, 10) for batch in (8, 16, 32, 64))
        triples.update((32, batch, 10) for batch in (4, 8, 16, 32))
        triples.update((48, batch, 10) for batch in (2, 4, 8, 16))
    else:
        # CPU is a representative fallback/RSS lane, not a duplicate 37-point scan.
        triples.update((cutoff, 2, 2) for cutoff in (8, 12, 16))
        triples.update((8, batch, 2) for batch in (1, 4, 8))
        triples.update((8, 4, horizon) for horizon in (2, 6, 10))
        triples.add((16, 2, 10))
    return tuple(
        TrainingPointConfig(cutoff=cutoff, batch_size=batch, full_cycles=horizon, **common)
        for cutoff, batch, horizon in sorted(triples)
    )


def validate_production_design(points: Sequence[TrainingPointConfig], device: str) -> None:
    if not points:
        raise ValueError("scan design must not be empty")
    if any(point.device != device for point in points):
        raise ValueError("all scan points must use the requested device")
    triples = {(point.cutoff, point.batch_size, point.full_cycles) for point in points}
    cutoffs = {point.cutoff for point in points}
    batches = {point.batch_size for point in points}
    horizons = {point.full_cycles for point in points}
    if device == "cuda":
        if not {8, 12, 16, 18, 24, 32, 48}.issubset(cutoffs):
            raise ValueError("CUDA production scan lacks the cutoff axis")
        if not {1, 4, 8, 16, 32, 64, 128, 256, 512}.issubset(batches):
            raise ValueError("CUDA production scan lacks the batch axis")
        if not set(range(2, 11)).issubset(horizons):
            raise ValueError("CUDA production scan must cover every 2--10 cycle horizon")
        required = {
            (16, batch, horizon)
            for batch in (8, 16)
            for horizon in range(2, 11)
        }
        if not required.issubset(triples):
            raise ValueError("CUDA production scan lacks the 2--10 envelope matrix")
        if not {(18, 4, 10), (24, 4, 10)}.issubset(triples):
            raise ValueError("CUDA production scan lacks high-cutoff long-horizon anchors")
        frontier = {
            (16, 512, 10),
            (16, 576, 10),
            (24, 64, 10),
            (32, 32, 10),
            (48, 16, 10),
        }
        if not frontier.issubset(triples):
            raise ValueError("CUDA production scan lacks isolated resource-frontier probes")
    else:
        if len(points) < 8 or max(cutoffs) < 16 or max(batches) < 8 or 10 not in horizons:
            raise ValueError("CPU production lane is too small to report RSS/runtime scaling")


def summarize_scan(results: Sequence[TrainingPointResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("results must not be empty")
    devices = sorted({result.device for result in results})
    primary_device = "cuda" if "cuda" in devices else devices[0]
    primary = [result for result in results if result.device == primary_device]
    largest_batch_by_horizon: dict[str, int | None] = {}
    preferred_batch_by_horizon: dict[str, int | None] = {}
    for horizon in range(2, 11):
        candidates = [
            result.batch_size
            for result in primary
            if result.cutoff == 16
            and result.full_cycles == horizon
            and result.feasible
        ]
        preferred_candidates = [
            result.batch_size
            for result in primary
            if result.cutoff == 16
            and result.full_cycles == horizon
            and result.preferred
        ]
        largest_batch_by_horizon[str(horizon)] = max(candidates) if candidates else None
        preferred_batch_by_horizon[str(horizon)] = (
            max(preferred_candidates) if preferred_candidates else None
        )
    values = list(largest_batch_by_horizon.values())
    all_horizons_observed = all(value is not None for value in values)
    common_batch = min(value for value in values if value is not None) if all_horizons_observed else None
    high_cutoff_long_horizon = [
        result
        for result in primary
        if result.cutoff >= 18
        and result.full_cycles == 10
        and result.batch_size >= 4
        and result.feasible
    ]
    envelope_confirmed = bool(
        all_horizons_observed
        and common_batch is not None
        and common_batch >= 8
        and high_cutoff_long_horizon
    )
    fatal_statuses = {"exception", "timeout", "numerical_failure"}
    fatal = [result.point_id for result in results if result.status in fatal_statuses]
    resource_statuses = {"oom", "preflight_rejected", "memory_exceeded", "runtime_exceeded"}
    resource_boundary = [
        result.point_id for result in results if result.status in resource_statuses
    ]
    decision = (
        "FEASIBLE_2_TO_10_CYCLE_TEACHER_KERNEL"
        if envelope_confirmed
        else "FALSIFIED_AT_REGISTERED_RESOURCE_GATE"
    )
    return {
        "primary_device": primary_device,
        "devices": devices,
        "point_count": len(results),
        "status_counts": {
            status: sum(result.status == status for result in results)
            for status in sorted({result.status for result in results})
        },
        "largest_tested_feasible_batch_by_horizon_at_cutoff16": largest_batch_by_horizon,
        "largest_tested_preferred_batch_by_horizon_at_cutoff16": preferred_batch_by_horizon,
        "common_feasible_batch_for_cycles_2_to_10": common_batch,
        "high_cutoff_long_horizon_points": [result.point_id for result in high_cutoff_long_horizon],
        "two_to_ten_cycle_envelope_confirmed": envelope_confirmed,
        "fatal_point_ids": fatal,
        "resource_boundary_point_ids": resource_boundary,
        "resource_frontier_observed": bool(resource_boundary),
        "decision": decision,
        "claim_boundary": (
            "The decision covers one current-host forward/backward/Adam-update kernel. It does not "
            "prove optimization convergence, seed robustness, lifetime ranking, physical "
            "cutoff convergence, device calibration, or FPGA timing."
        ),
    }


def run_feasibility_scan(
    *,
    devices: Sequence[str] = ("cuda", "cpu"),
    output: str | Path | None = None,
    warmup_steps: int = 1,
    repeats: int = 3,
    runtime_budget_seconds: float = 10.0,
    preferred_runtime_seconds: float = 2.0,
    timeout_seconds: float = 900.0,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the preregistered subprocess-isolated production scan."""

    th, _ = _require_runtime()
    normalized_devices = tuple(dict.fromkeys(devices))
    if not normalized_devices or any(device not in {"cpu", "cuda"} for device in normalized_devices):
        raise ValueError("devices must contain cpu and/or cuda")
    if "cuda" in normalized_devices and not th.cuda.is_available():
        raise RuntimeError("CUDA production lane requested but unavailable")
    points: list[TrainingPointConfig] = []
    for device in normalized_devices:
        designed = default_scan_points(
            device,
            warmup_steps=warmup_steps,
            repeats=repeats,
            runtime_budget_seconds=runtime_budget_seconds,
            preferred_runtime_seconds=preferred_runtime_seconds,
        )
        validate_production_design(designed, device)
        points.extend(designed)
    reusable: dict[str, TrainingPointResult] = {}
    output_path = Path(output) if output is not None else None
    if resume and output_path is not None and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("task_id") != "T2.3.6" or previous.get("scope") != DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE:
            raise ValueError("resume artifact has the wrong task or scope")
        contract = previous.get("measurement_contract", {})
        expected_contract = {
            "warmup_steps": warmup_steps,
            "timed_repeats": repeats,
            "runtime_budget_seconds_per_step": runtime_budget_seconds,
            "preferred_runtime_seconds_per_step": preferred_runtime_seconds,
            "optimizer": "Adam",
            "learning_rate": 1.0e-4,
        }
        for key, expected in expected_contract.items():
            if contract.get(key) != expected:
                raise ValueError(f"resume artifact measurement contract mismatch: {key}")
        for item in previous.get("points", []):
            result = TrainingPointResult(**item)
            reusable[result.point_id] = result
    results: list[TrainingPointResult] = []
    reused_count = 0
    new_count = 0
    for point in points:
        previous_result = reusable.get(point.point_id)
        if (
            previous_result is not None
            and previous_result.warmup_steps == point.warmup_steps
            and previous_result.repeats == point.repeats
            and previous_result.runtime_budget_seconds == point.runtime_budget_seconds
            and previous_result.preferred_runtime_seconds == point.preferred_runtime_seconds
            and previous_result.maximum_memory_fraction == point.maximum_memory_fraction
            and previous_result.grid_points == point.grid_points
            and previous_result.score_baseline == point.score_baseline
            and previous_result.learning_rate == point.learning_rate
            and previous_result.seed == point.seed
        ):
            results.append(previous_result)
            reused_count += 1
        else:
            results.append(run_point_subprocess(point, timeout_seconds=timeout_seconds))
            new_count += 1
    summary = summarize_scan(results)
    checks = {
        "production_design_has_cuda_cutoff_batch_and_every_2_to_10_horizon": (
            "cuda" not in normalized_devices
            or len(default_scan_points("cuda")) >= 50
        ),
        "cpu_rss_fallback_lane_present": "cpu" in normalized_devices,
        "all_points_have_repeated_measurements": all(result.repeats >= 3 for result in results),
        "policy_has_fifteen_outputs": RecurrentPolicySpec().output_controls == 15,
        "policy_parameter_count_matches_gru10_mlp256": (
            RecurrentPolicySpec().analytic_parameter_count == 72913
        ),
        "all_completed_points_have_finite_objective_and_gradients": all(
            result.objective_finite and result.gradients_finite
            for result in results
            if result.status not in {"exception", "timeout", "oom", "preflight_rejected"}
        ),
        "all_completed_points_pass_density_numerics": all(
            result.numerical_stable
            for result in results
            if result.status not in {"exception", "timeout", "oom", "preflight_rejected"}
        ),
        "memory_is_measured_not_only_analytic": all(
            result.peak_rss_bytes is not None
            and (result.device != "cuda" or result.cuda_peak_allocated_bytes is not None)
            for result in results
            if result.status not in {"exception", "timeout", "oom", "preflight_rejected"}
        ),
        "scan_yields_feasibility_or_registered_falsification": summary["decision"] in {
            "FEASIBLE_2_TO_10_CYCLE_TEACHER_KERNEL",
            "FALSIFIED_AT_REGISTERED_RESOURCE_GATE",
        },
        "isolated_resource_frontier_is_observed": (
            "cuda" not in normalized_devices or summary["resource_frontier_observed"]
        ),
        "runtime_frontier_is_observed": (
            "cuda" not in normalized_devices
            or any(result.status == "runtime_exceeded" for result in results)
        ),
        "memory_frontier_is_observed": (
            "cuda" not in normalized_devices
            or any(
                result.status in {"oom", "memory_exceeded", "preflight_rejected"}
                or (
                    result.observed_memory_fraction is not None
                    and result.observed_memory_fraction >= 0.70
                )
                for result in results
                if result.device == "cuda"
            )
        ),
        "no_unisolated_worker_failure": not summary["fatal_point_ids"],
    }
    policy_payload = asdict(RecurrentPolicySpec())
    policy_payload["analytic_parameter_count"] = RecurrentPolicySpec().analytic_parameter_count
    payload = {
        "schema_version": "1.0",
        "task_id": "T2.3.6",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_id": SBS_PROTOCOL_ID,
        "scope": DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE,
        "policy_spec": policy_payload,
        "measurement_contract": {
            "workload": "causal 15-output recurrent policy + sampled trajectory + reward/score backward + Adam update",
            "warmup_steps": warmup_steps,
            "timed_repeats": repeats,
            "runtime_statistic": "median with p90 and raw repeats",
            "cpu_memory": "2 ms process-RSS peak sampler in an isolated worker",
            "cuda_memory": "torch max allocated/reserved bytes including backward graph",
            "runtime_budget_seconds_per_step": runtime_budget_seconds,
            "preferred_runtime_seconds_per_step": preferred_runtime_seconds,
            "maximum_memory_fraction": 0.75,
            "real_dtype": "float64",
            "grid_points": 2049,
            "constant_score_baseline": 0.35,
            "optimizer": "Adam",
            "learning_rate": 1.0e-4,
            "subprocess_isolation": True,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "logical_cpu_count": os.cpu_count(),
            "torch": th.__version__,
            "cuda_runtime": th.version.cuda,
            "cuda_available": bool(th.cuda.is_available()),
            "cuda_device": (
                th.cuda.get_device_name(0) if th.cuda.is_available() else None
            ),
        },
        "execution": {
            "resume_enabled": bool(resume),
            "reused_contract_identical_points": reused_count,
            "newly_executed_points": new_count,
        },
        "summary": summary,
        "checks": checks,
        "points": [asdict(result) for result in results],
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return payload


def _config_from_args(args: argparse.Namespace) -> TrainingPointConfig:
    return TrainingPointConfig(
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        full_cycles=args.full_cycles,
        device=args.device,
        real_dtype=args.real_dtype,
        grid_points=args.grid_points,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        score_baseline=args.score_baseline,
        learning_rate=args.learning_rate,
        runtime_budget_seconds=args.runtime_budget_seconds,
        preferred_runtime_seconds=args.preferred_runtime_seconds,
        maximum_memory_fraction=args.maximum_memory_fraction,
        seed=args.seed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--full-cycles", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=("cuda", "cpu"))
    parser.add_argument("--real-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--grid-points", type=int, default=2049)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--score-baseline", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--runtime-budget-seconds", type=float, default=10.0)
    parser.add_argument("--preferred-runtime-seconds", type=float, default=2.0)
    parser.add_argument("--maximum-memory-fraction", type=float, default=0.75)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        result = safe_benchmark_training_point(_config_from_args(args))
        print(RESULT_MARKER + json.dumps(asdict(result), ensure_ascii=False))
        return 0
    payload = run_feasibility_scan(
        devices=args.devices,
        output=args.output,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        runtime_budget_seconds=args.runtime_budget_seconds,
        preferred_runtime_seconds=args.preferred_runtime_seconds,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
