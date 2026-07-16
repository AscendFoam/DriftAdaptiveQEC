"""T3.2.8 autonomous versus measurement-feedback sBs wall-clock benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from physics.autonomous_sbs import (
    AUTONOMOUS_TIMING,
    MEASUREMENT_TIMING,
    MODEL_SCOPE,
    PAPER_SOURCE,
    NonselectiveSBSConfig,
    NonselectiveSBSSimulator,
    finite_horizon_area_lifetime,
    validate_timing_contract,
)
from physics.differentiable_sbs_trajectory import (
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


TASK_ID = "T3.2.8"
CONTRACT_ID = "T328-PROTOCOL-NATIVE-COMMON-700US-V1"
DEFAULT_ARTIFACT = Path("docs/t3_2_8_autonomous_sbs_wallclock_validation.json")
DEFAULT_SOURCE_DATA = Path("docs/t3_2_8_autonomous_sbs_wallclock_source_data.csv")
PAPER_PATH = Path(PAPER_SOURCE)


NOISE_PROFILES: Mapping[str, tuple[float, float, float]] = {
    # Puviani Table S5 in microseconds for a 10 us standard cycle.
    "high": (245.0, 50.0, 60.0),
    "medium": (490.0, 100.0, 120.0),
    "low": (610.0, 280.0, 238.0),
}


@dataclass(frozen=True)
class WallClockBenchmarkConfig:
    common_horizon_us: int = 700
    cutoffs: tuple[int, ...] = (12, 16)
    projector_delta: float = 0.34
    device: str = "cuda"
    real_dtype: str = "float64"

    def __post_init__(self) -> None:
        if isinstance(self.common_horizon_us, bool) or int(self.common_horizon_us) <= 0:
            raise ValueError("common_horizon_us must be a positive integer")
        object.__setattr__(self, "common_horizon_us", int(self.common_horizon_us))
        cutoffs = tuple(int(value) for value in self.cutoffs)
        if len(cutoffs) < 2 or len(set(cutoffs)) != len(cutoffs):
            raise ValueError("at least two unique cutoff lanes are required")
        if any(not 4 <= value <= 48 for value in cutoffs):
            raise ValueError("cutoffs must lie in [4,48]")
        object.__setattr__(self, "cutoffs", cutoffs)
        for duration in (
            MEASUREMENT_TIMING.full_cycle_duration_ns / 1000.0,
            AUTONOMOUS_TIMING.full_cycle_duration_ns / 1000.0,
        ):
            if self.common_horizon_us % int(duration) != 0:
                raise ValueError("common horizon must contain an integer number of both cycles")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype != "float64":
            raise ValueError("production wall-clock comparison requires float64")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("T3.2.8 requires the local DLEnv PyTorch environment")
    return torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "autonomous_sbs.py",
        Path(__file__).resolve().parents[2] / "physics" / "differentiable_sbs_trajectory.py",
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def nonselective_measurement_equivalence_audit(*, cutoff: int = 6) -> dict[str, float | bool]:
    """Compare exact nonselective propagation with enumeration of all g/e branches."""

    th = _require_torch()
    branches = th.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=th.int64)
    sampled = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(cutoff=cutoff, full_cycles=1, batch_size=4)
    ).run(forced_outcomes=branches, seed=328)
    weighted = th.einsum(
        "b,bij->ij",
        sampled.trajectory_probability.to(sampled.final_cavity_density.dtype),
        sampled.final_cavity_density,
    )
    exact = NonselectiveSBSSimulator(
        NonselectiveSBSConfig(
            mode="measurement_feedback", full_cycles=1, cutoff=cutoff
        )
    ).run()
    probability_sum = float(sampled.trajectory_probability.sum().detach().cpu())
    difference = float(th.max(th.abs(weighted - exact.final_cavity_density)).detach().cpu())
    return {
        "branch_probability_sum": probability_sum,
        "maximum_density_difference": difference,
        "all_four_branches_positive": bool(
            th.all(sampled.trajectory_probability > 0.0).detach().cpu()
        ),
        "passes": abs(probability_sum - 1.0) <= 2.0e-12 and difference <= 2.0e-12,
    }


def zero_noise_duration_invariance_audit(*, cutoff: int = 6) -> dict[str, float | bool]:
    """With effectively infinite lifetimes, duration alone must not change a cycle map."""

    results = []
    for mode in ("measurement_feedback", "autonomous"):
        results.append(
            NonselectiveSBSSimulator(
                NonselectiveSBSConfig(
                    mode=mode,
                    full_cycles=2,
                    cutoff=cutoff,
                    cavity_lifetime_us=1.0e15,
                    ancilla_t1_us=1.0e15,
                    ancilla_t2_us=1.0e15,
                )
            ).run()
        )
    difference = float(
        torch.max(
            torch.abs(results[0].final_cavity_density - results[1].final_cavity_density)
        ).detach().cpu()
    )
    return {"maximum_density_difference": difference, "passes": difference <= 2.0e-12}


def _protocol_cycles(config: WallClockBenchmarkConfig, mode: str) -> int:
    duration = (
        MEASUREMENT_TIMING.full_cycle_duration_ns
        if mode == "measurement_feedback"
        else AUTONOMOUS_TIMING.full_cycle_duration_ns
    ) / 1000.0
    return int(round(config.common_horizon_us / duration))


def _run_lane(
    config: WallClockBenchmarkConfig, *, cutoff: int, noise_name: str, mode: str
) -> dict[str, Any]:
    cavity, ancilla_t1, ancilla_t2 = NOISE_PROFILES[noise_name]
    result = NonselectiveSBSSimulator(
        NonselectiveSBSConfig(
            mode=mode,
            full_cycles=_protocol_cycles(config, mode),
            cutoff=cutoff,
            projector_delta=config.projector_delta,
            cavity_lifetime_us=cavity,
            ancilla_t1_us=ancilla_t1,
            ancilla_t2_us=ancilla_t2,
            device=config.device,
            real_dtype=config.real_dtype,
        )
    ).run()
    payload = result.to_dict()
    payload["noise_profile"] = noise_name
    payload["metrics"] = {
        "fidelity": finite_horizon_area_lifetime(result.time_us, result.fidelity),
        "logical_z_signal": finite_horizon_area_lifetime(
            result.time_us, result.logical_z_signal
        ),
        "final_fidelity_at_common_horizon": float(result.fidelity[-1]),
        "final_logical_z_at_common_horizon": float(result.logical_z_signal[-1]),
        "final_code_survival_at_common_horizon": float(result.code_survival[-1]),
    }
    return payload


def _comparison(measurement: Mapping[str, Any], autonomous: Mapping[str, Any]) -> dict[str, float]:
    m = measurement["metrics"]
    a = autonomous["metrics"]
    m_events = measurement["event_accounting"]
    a_events = autonomous["event_accounting"]
    return {
        "autonomous_minus_measurement_final_fidelity": float(
            a["final_fidelity_at_common_horizon"] - m["final_fidelity_at_common_horizon"]
        ),
        "autonomous_minus_measurement_final_logical_z": float(
            a["final_logical_z_at_common_horizon"] - m["final_logical_z_at_common_horizon"]
        ),
        "autonomous_to_measurement_logical_lifetime_us_ratio": float(
            a["logical_z_signal"]["area_equivalent_lifetime_us"]
            / m["logical_z_signal"]["area_equivalent_lifetime_us"]
        ),
        "autonomous_to_measurement_logical_lifetime_protocol_cycle_ratio": float(
            a["logical_z_signal"]["area_equivalent_lifetime_protocol_cycles"]
            / m["logical_z_signal"]["area_equivalent_lifetime_protocol_cycles"]
        ),
        "autonomous_to_measurement_reset_rate_ratio": float(
            a_events["resets_per_100us"] / m_events["resets_per_100us"]
        ),
        "autonomous_to_measurement_active_gate_rate_ratio": float(
            a_events["active_gates_per_100us"] / m_events["active_gates_per_100us"]
        ),
        "measurement_events_avoided_at_common_horizon": float(
            m_events["measurement_events"] - a_events["measurement_events"]
        ),
        "additional_autonomous_resets_at_common_horizon": float(
            a_events["reset_events"] - m_events["reset_events"]
        ),
        "additional_autonomous_active_gates_at_common_horizon": float(
            a_events["active_gate_applications"] - m_events["active_gate_applications"]
        ),
    }


def _source_rows(lanes: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane_id, lane in lanes.items():
        cutoff = lane["cutoff"]
        noise = lane["noise_profile"]
        for mode in ("measurement_feedback", "autonomous"):
            protocol = lane[mode]
            for index, time_us in enumerate(protocol["time_us"]):
                for metric in (
                    "fidelity", "code_survival", "logical_z_signal", "conditional_logical_z"
                ):
                    rows.append(
                        {
                            "row_type": "curve",
                            "lane_id": lane_id,
                            "cutoff": cutoff,
                            "noise_profile": noise,
                            "mode": mode,
                            "cycle": index,
                            "time_us": time_us,
                            "metric": metric,
                            "value": protocol[metric][index],
                        }
                    )
            for metric, value in protocol["event_accounting"].items():
                rows.append(
                    {
                        "row_type": "event_accounting",
                        "lane_id": lane_id,
                        "cutoff": cutoff,
                        "noise_profile": noise,
                        "mode": mode,
                        "cycle": "",
                        "time_us": protocol["event_accounting"]["total_physical_time_us"],
                        "metric": metric,
                        "value": value,
                    }
                )
        for metric, value in lane["comparison"].items():
            rows.append(
                {
                    "row_type": "comparison",
                    "lane_id": lane_id,
                    "cutoff": cutoff,
                    "noise_profile": noise,
                    "mode": "paired",
                    "cycle": "",
                    "time_us": lane["measurement_feedback"]["event_accounting"]["total_physical_time_us"],
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "row_type", "lane_id", "cutoff", "noise_profile", "mode",
                "cycle", "time_us", "metric", "value",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_benchmark(
    config: WallClockBenchmarkConfig = WallClockBenchmarkConfig(),
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    th = _require_torch()
    if config.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA production run requested but unavailable")
    started = time.perf_counter()
    source_text = PAPER_PATH.read_text(encoding="utf-8")
    required_fragments = (
        "total duration $ \\tau_{\\mathrm{cycle}}=10\\mu\\mathrm{s} $",
        "duration of half-cycle for the autonomous QEC will be 0.35",
        "full cycle of the autonomous QEC lasts 0.7",
        "ancillary qubit is still reset after layer 4",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source_text]
    if missing:
        raise ValueError(f"paper timing source fragments are missing: {missing}")

    equivalence = nonselective_measurement_equivalence_audit()
    zero_noise = zero_noise_duration_invariance_audit()
    lanes: dict[str, Any] = {}
    for cutoff in config.cutoffs:
        for noise_name in NOISE_PROFILES:
            lane_id = f"cutoff{cutoff}_{noise_name}"
            measurement = _run_lane(
                config, cutoff=cutoff, noise_name=noise_name, mode="measurement_feedback"
            )
            autonomous = _run_lane(
                config, cutoff=cutoff, noise_name=noise_name, mode="autonomous"
            )
            lanes[lane_id] = {
                "cutoff": cutoff,
                "noise_profile": noise_name,
                "measurement_feedback": measurement,
                "autonomous": autonomous,
                "comparison": _comparison(measurement, autonomous),
            }

    timing_gates = dict(validate_timing_contract())
    all_protocols = [
        lane[mode]
        for lane in lanes.values()
        for mode in ("measurement_feedback", "autonomous")
    ]
    gates = {
        **timing_gates,
        "paper_source_fragments_and_hash_are_live": not missing and len(_sha256(PAPER_PATH)) == 64,
        "nonselective_measurement_matches_all_branch_enumeration": bool(equivalence["passes"]),
        "duration_difference_vanishes_without_dissipation": bool(zero_noise["passes"]),
        "all_lanes_share_exact_common_wallclock_horizon": all(
            item["event_accounting"]["total_physical_time_us"] == config.common_horizon_us
            and item["time_us"][-1] == config.common_horizon_us
            for item in all_protocols
        ),
        "measurement_and_reset_event_counts_are_protocol_native": all(
            lane["measurement_feedback"]["event_accounting"]["measurement_events"]
            == lane["measurement_feedback"]["event_accounting"]["reset_events"]
            and lane["autonomous"]["event_accounting"]["measurement_events"] == 0
            and lane["autonomous"]["event_accounting"]["reset_events"]
            > lane["measurement_feedback"]["event_accounting"]["reset_events"]
            for lane in lanes.values()
        ),
        "reset_and_active_gate_rate_ratio_is_exactly_ten_over_seven": all(
            abs(lane["comparison"][metric] - 10.0 / 7.0) <= 2.0e-12
            for lane in lanes.values()
            for metric in (
                "autonomous_to_measurement_reset_rate_ratio",
                "autonomous_to_measurement_active_gate_rate_ratio",
            )
        ),
        "all_density_and_lifetime_metrics_are_finite": all(
            item["maximum_trace_error"] <= 2.0e-9
            and item["maximum_hermiticity_error"] <= 2.0e-9
            and item["minimum_final_eigenvalue"] >= -2.0e-8
            and all(
                np.isfinite(item["metrics"][metric]["area_equivalent_lifetime_us"])
                for metric in ("fidelity", "logical_z_signal")
            )
            for item in all_protocols
        ),
        "both_cutoffs_and_all_noise_profiles_are_present": (
            len(lanes) == len(config.cutoffs) * len(NOISE_PROFILES)
            and {lane["noise_profile"] for lane in lanes.values()} == set(NOISE_PROFILES)
        ),
        "no_desired_performance_direction_is_required": True,
        "timing_and_runtime_are_not_target_hardware_measurements": all(
            not item["config"]["timing"]["target_hardware_measured"]
            for item in all_protocols
        ),
    }
    required_gates = tuple(gates)
    status = "PASS" if all(gates[name] for name in required_gates) else "FAIL"
    rows = _source_rows(lanes)
    _write_csv(rows, source_data_path)
    artifact: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": status,
        "scope": MODEL_SCOPE,
        "contract_id": CONTRACT_ID,
        "implementation_sha256": implementation_sha256(),
        "config": asdict(config),
        "noise_profiles_us": {
            name: {"cavity_lifetime": values[0], "ancilla_t1": values[1], "ancilla_t2": values[2]}
            for name, values in NOISE_PROFILES.items()
        },
        "literature": {
            "source_path": PAPER_PATH.as_posix(),
            "source_sha256": _sha256(PAPER_PATH),
            "anchors": ["451-459", "597-599", "623"],
            "timing_is_literature_simulation_not_target_hardware": True,
        },
        "timing_profiles": {
            "measurement_feedback": asdict(MEASUREMENT_TIMING),
            "autonomous": asdict(AUTONOMOUS_TIMING),
        },
        "method_audits": {
            "nonselective_measurement_equivalence": equivalence,
            "zero_noise_duration_invariance": zero_noise,
        },
        "lanes": lanes,
        "gates": gates,
        "required_gates": list(required_gates),
        "source_data": {
            "path": source_data_path.as_posix(),
            "sha256": _sha256(source_data_path),
            "row_count": len(rows),
        },
        "workload": {
            "lane_count": len(lanes),
            "total_full_cycles": sum(
                int(item["event_accounting"]["full_cycles"])
                for item in all_protocols
            ),
            "deterministic_nonselective_no_monte_carlo_ci": True,
        },
        "claim_boundary": {
            "allowed": "finite-cutoff nominal-control model comparison per protocol cycle and common literature wall-clock time",
            "forbidden": [
                "paper Fig.3b numerical reproduction",
                "trained autonomous optimum",
                "multilevel leakage or pulse dynamics",
                "target-board or device measured timing",
            ],
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    _atomic_json(artifact, artifact_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    artifact = run_benchmark(
        WallClockBenchmarkConfig(device=args.device),
        artifact_path=args.artifact,
        source_data_path=args.source_data,
    )
    summary = {
        lane_id: lane["comparison"]
        for lane_id, lane in artifact["lanes"].items()
    }
    print(json.dumps({"task_id": TASK_ID, "status": artifact["status"], "comparison": summary, "gates": artifact["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

