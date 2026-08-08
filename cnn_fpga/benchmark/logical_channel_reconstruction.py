"""T5.3.1 six-state matched QEC-on/off logical-channel reconstruction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cnn_fpga.benchmark.autonomous_sbs_wallclock_baseline import (
    implementation_sha256 as parent_implementation_sha256,
)
from physics.autonomous_sbs import MEASUREMENT_TIMING
from physics.fock_logical_channel import (
    AXIS_LABELS,
    MODEL_SCOPE,
    PAULI_LABELS,
    STATE_LABELS,
    FockLogicalChannelConfig,
    FockLogicalChannelSimulator,
    finite_horizon_pauli_lifetime,
    reconstruct_code_subchannel,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


TASK_ID = "T5.3.1"
CONTRACT_ID = "T531-SIX-STATE-MATCHED-CPTNI-CHANNEL-V1"
DEFAULT_ARTIFACT = Path("docs/t5_3_1_logical_channel_reconstruction.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_3_1_logical_channel_source_data.csv")
PARENT_ARTIFACT = Path("docs/t3_2_8_autonomous_sbs_wallclock_validation.json")

NOISE_PROFILES: Mapping[str, tuple[float, float, float]] = {
    "high": (245.0, 50.0, 60.0),
    "medium": (490.0, 100.0, 120.0),
    "low": (610.0, 280.0, 238.0),
}


@dataclass(frozen=True)
class LogicalChannelBenchmarkConfig:
    full_cycles: int = 30
    cutoffs: tuple[int, ...] = (12, 24, 36, 40)
    projector_delta: float = 0.34
    device: str = "cuda"
    real_dtype: str = "float64"

    def __post_init__(self) -> None:
        if isinstance(self.full_cycles, bool) or int(self.full_cycles) < 3:
            raise ValueError("full_cycles must be an integer >=3")
        object.__setattr__(self, "full_cycles", int(self.full_cycles))
        cutoffs = tuple(int(value) for value in self.cutoffs)
        if len(cutoffs) != 4 or len(set(cutoffs)) != 4:
            raise ValueError("exactly four unique cutoff lanes are required")
        if any(not 4 <= value <= 48 for value in cutoffs):
            raise ValueError("cutoffs must lie in [4,48]")
        if tuple(sorted(cutoffs)) != cutoffs:
            raise ValueError("cutoffs must be strictly increasing")
        object.__setattr__(self, "cutoffs", cutoffs)
        if not np.isfinite(self.projector_delta) or self.projector_delta <= 0.0:
            raise ValueError("projector_delta must be finite and positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.real_dtype != "float64":
            raise ValueError("formal reconstruction requires float64")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "fock_logical_channel.py",
        Path(__file__).resolve().parents[2] / "physics" / "autonomous_sbs.py",
        Path(__file__).resolve().parents[2] / "physics" / "differentiable_sbs_trajectory.py",
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _parent_audit() -> dict[str, Any]:
    payload = json.loads(PARENT_ARTIFACT.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    return {
        "path": PARENT_ARTIFACT.as_posix(),
        "sha256": _sha256(PARENT_ARTIFACT),
        "task_id": payload.get("task_id"),
        "status": payload.get("status"),
        "all_required_gates_passed": all(
            gates.get(name) is True for name in payload.get("required_gates", [])
        ),
        "stored_implementation_sha256": payload.get("implementation_sha256"),
        "live_implementation_sha256": parent_implementation_sha256(),
        "implementation_hash_matches": (
            payload.get("implementation_sha256") == parent_implementation_sha256()
        ),
    }


def _lane_id(cutoff: int, noise: str, mode: str) -> str:
    return f"cutoff{cutoff}:{noise}:{mode}"


def _run_lane(
    config: LogicalChannelBenchmarkConfig,
    *,
    cutoff: int,
    noise: str,
    mode: str,
) -> dict[str, Any]:
    cavity, ancilla_t1, ancilla_t2 = NOISE_PROFILES[noise]
    result = FockLogicalChannelSimulator(
        FockLogicalChannelConfig(
            mode=mode,
            full_cycles=config.full_cycles,
            cutoff=cutoff,
            projector_delta=config.projector_delta,
            cavity_lifetime_us=cavity,
            ancilla_t1_us=ancilla_t1,
            ancilla_t2_us=ancilla_t2,
            device=config.device,
            real_dtype=config.real_dtype,
        )
    ).run()
    payload = result.to_dict(include_projected_outputs=True)
    payload["lane_id"] = _lane_id(cutoff, noise, mode)
    payload["noise_profile"] = noise
    payload["final_summary"] = {
        "ptm": result.ptm[-1].tolist(),
        "mean_leakage": result.tomography[-1].mean_leakage,
        "survival_spread": result.tomography[-1].survival_spread,
        "off_diagonal_pauli_norm": result.tomography[-1].off_diagonal_pauli_norm,
        "coherent_rotation_norm": result.tomography[-1].coherent_rotation_norm,
        "nonunital_code_flow_norm": result.tomography[-1].nonunital_code_flow_norm,
        "state_dependent_survival_norm": result.tomography[-1].state_dependent_survival_norm,
    }
    return payload


def _matched_comparisons(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cutoff in sorted({int(lane["config"]["cutoff"]) for lane in lanes.values()}):
        for noise in NOISE_PROFILES:
            on = lanes[_lane_id(cutoff, noise, "qec_on")]
            off = lanes[_lane_id(cutoff, noise, "qec_off")]
            comparison: dict[str, Any] = {
                "cutoff": cutoff,
                "noise_profile": noise,
                "cycle_duration_us": float(on["config"]["cycle_duration_us"]),
                "matched_fields": [
                    "full_cycles",
                    "cutoff",
                    "projector_delta",
                    "cavity_lifetime_us",
                    "ancilla_t1_us",
                    "ancilla_t2_us",
                    "cycle_duration_us",
                    "device",
                    "real_dtype",
                    "scope",
                ],
                "only_intervention_difference": "mode:qec_on_vs_qec_off",
                "performance_direction_required": False,
                "axes": {},
                "qec_on_minus_off_final_mean_leakage": float(
                    on["final_summary"]["mean_leakage"]
                    - off["final_summary"]["mean_leakage"]
                ),
            }
            for axis in AXIS_LABELS:
                on_metric = on["pauli_lifetimes"][axis]
                off_metric = off["pauli_lifetimes"][axis]
                off_area = float(off_metric["truncated_signed_area_cycles"])
                comparison["axes"][axis] = {
                    "qec_on_area_cycles": float(on_metric["truncated_signed_area_cycles"]),
                    "qec_off_area_cycles": off_area,
                    "qec_on_minus_off_area_cycles": float(
                        on_metric["truncated_signed_area_cycles"] - off_area
                    ),
                    "qec_on_to_off_area_ratio": (
                        None
                        if abs(off_area) <= 1.0e-14
                        else float(on_metric["truncated_signed_area_cycles"] / off_area)
                    ),
                    "qec_on_e_fold_cycles": on_metric["e_fold_crossing_cycles"],
                    "qec_off_e_fold_cycles": off_metric["e_fold_crossing_cycles"],
                }
            rows.append(comparison)
    return rows


def _cutoff_diagnostics(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    cutoffs = sorted({int(lane["config"]["cutoff"]) for lane in lanes.values()})
    rows = []
    for lower, higher in zip(cutoffs[:-1], cutoffs[1:], strict=True):
        for noise in NOISE_PROFILES:
            for mode in ("qec_off", "qec_on"):
                low_lane = lanes[_lane_id(lower, noise, mode)]
                high_lane = lanes[_lane_id(higher, noise, mode)]
                low_ptm = np.asarray(low_lane["final_summary"]["ptm"], dtype=np.float64)
                high_ptm = np.asarray(high_lane["final_summary"]["ptm"], dtype=np.float64)
                rows.append(
                    {
                        "noise_profile": noise,
                        "mode": mode,
                        "lower_cutoff": lower,
                        "higher_cutoff": higher,
                        "terminal_registered_pair": higher == cutoffs[-1],
                        "final_ptm_frobenius_difference": float(
                            np.linalg.norm(high_ptm - low_ptm, ord="fro")
                        ),
                        "final_mean_leakage_absolute_difference": abs(
                            float(high_lane["final_summary"]["mean_leakage"])
                            - float(low_lane["final_summary"]["mean_leakage"])
                        ),
                        "used_as_infinite_cutoff_convergence_claim": False,
                    }
                )
    return rows


def _deep_close(left: Any, right: Any, *, tolerance: float = 2.0e-9) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _deep_close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _deep_close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if left is None or right is None or isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
        return left == right
    try:
        return bool(np.isclose(float(left), float(right), rtol=2.0e-9, atol=tolerance))
    except (TypeError, ValueError):
        return left == right


def _raw_reconstruction_consistency(lanes: Mapping[str, Mapping[str, Any]]) -> bool:
    """Recompute every derived channel quantity from raw six-state 2x2 outputs."""

    try:
        for lane in lanes.values():
            raw = np.asarray(lane["projected_output_real"], dtype=np.float64) + 1.0j * np.asarray(
                lane["projected_output_imag"], dtype=np.float64
            )
            if raw.shape != (len(lane["cycles"]), len(STATE_LABELS), 2, 2):
                return False
            survival = np.trace(raw, axis1=-2, axis2=-1).real
            leakage = 1.0 - survival
            if not np.allclose(survival, lane["survival"], rtol=2.0e-9, atol=2.0e-9):
                return False
            if not np.allclose(leakage, lane["leakage"], rtol=2.0e-9, atol=2.0e-9):
                return False
            reconstructed = []
            conditional = np.empty((raw.shape[0], raw.shape[1], 3), dtype=np.float64)
            for cycle_index in range(raw.shape[0]):
                point = reconstruct_code_subchannel(
                    {
                        label: raw[cycle_index, state_index]
                        for state_index, label in enumerate(STATE_LABELS)
                    }
                )
                reconstructed.append(point)
                if not _deep_close(point.to_dict(), lane["tomography"][cycle_index]):
                    return False
                for state_index in range(raw.shape[1]):
                    weight = float(survival[cycle_index, state_index])
                    if weight <= 1.0e-14:
                        return False
                    conditional[cycle_index, state_index] = [
                        float(np.trace(pauli @ raw[cycle_index, state_index]).real / weight)
                        for pauli in (
                            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
                            np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
                            np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
                        )
                    ]
            if not np.allclose(
                conditional,
                lane["conditional_bloch_xyz"],
                rtol=2.0e-9,
                atol=2.0e-9,
            ):
                return False
            ptm = np.stack([point.ptm for point in reconstructed])
            cycles = np.asarray(lane["cycles"], dtype=np.float64)
            times = np.asarray(lane["time_us"], dtype=np.float64)
            for index, axis in enumerate(AXIS_LABELS, start=1):
                metric = finite_horizon_pauli_lifetime(cycles, times, ptm[:, index, index])
                if not _deep_close(metric, lane["pauli_lifetimes"][axis]):
                    return False
            final = reconstructed[-1]
            expected_final = {
                "ptm": final.ptm.tolist(),
                "mean_leakage": final.mean_leakage,
                "survival_spread": final.survival_spread,
                "off_diagonal_pauli_norm": final.off_diagonal_pauli_norm,
                "coherent_rotation_norm": final.coherent_rotation_norm,
                "nonunital_code_flow_norm": final.nonunital_code_flow_norm,
                "state_dependent_survival_norm": final.state_dependent_survival_norm,
            }
            if not _deep_close(expected_final, lane["final_summary"]):
                return False
    except (KeyError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
        return False
    return True


def _semantic_gates(payload: Mapping[str, Any]) -> dict[str, bool]:
    lanes = payload.get("lanes", {})
    parent = payload.get("parent_audit", {})
    expected_ids = {
        _lane_id(cutoff, noise, mode)
        for cutoff in payload.get("config", {}).get("cutoffs", [])
        for noise in NOISE_PROFILES
        for mode in ("qec_off", "qec_on")
    }
    lane_ids_match = set(lanes) == expected_ids and len(lanes) == 24
    all_tomography = [
        point for lane in lanes.values() for point in lane.get("tomography", [])
    ]
    all_lifetimes = [
        metric
        for lane in lanes.values()
        for metric in lane.get("pauli_lifetimes", {}).values()
    ]
    matched = True
    for cutoff in payload.get("config", {}).get("cutoffs", []):
        for noise in NOISE_PROFILES:
            on = lanes.get(_lane_id(cutoff, noise, "qec_on"), {}).get("config", {})
            off = lanes.get(_lane_id(cutoff, noise, "qec_off"), {}).get("config", {})
            if not on or not off:
                matched = False
                continue
            on_copy, off_copy = dict(on), dict(off)
            on_copy.pop("mode", None)
            off_copy.pop("mode", None)
            matched &= on_copy == off_copy
    initial_identity = max(
        (
            float(np.max(np.abs(np.asarray(lane["tomography"][0]["ptm"]) - np.eye(4))))
            for lane in lanes.values()
        ),
        default=float("inf"),
    )
    initial_leakage = max(
        (abs(float(lane["tomography"][0]["mean_leakage"])) for lane in lanes.values()),
        default=float("inf"),
    )
    event_ok = all(
        (
            lane["event_accounting"]["measurement_events"]
            == (2 * lane["config"]["full_cycles"] if lane["config"]["mode"] == "qec_on" else 0)
            and lane["event_accounting"]["reset_events"]
            == (2 * lane["config"]["full_cycles"] if lane["config"]["mode"] == "qec_on" else 0)
            and lane["event_accounting"]["active_gate_applications"]
            == (18 * lane["config"]["full_cycles"] if lane["config"]["mode"] == "qec_on" else 0)
            and lane["event_accounting"]["discarded_trajectories"] == 0
            and lane["event_accounting"]["postselected_trajectories"] == 0
        )
        for lane in lanes.values()
    ) if lanes else False
    lifetime_units = all(
        abs(
            float(metric["truncated_signed_area_us"])
            - 10.0 * float(metric["truncated_signed_area_cycles"])
        ) <= 2.0e-9
        and metric.get("definition", "").endswith("no exponential fit or postselection")
        for metric in all_lifetimes
    ) if all_lifetimes else False
    conditional_not_source = payload.get("tomography_contract", {}).get(
        "ptm_reconstruction_source"
    ) == "unnormalized_code_space_outputs"
    excluded = payload.get("evidence_routing", {}).get("excluded_heterogeneous_inputs", [])
    return {
        "parent_timing_channel_artifact_is_live": bool(
            parent.get("task_id") == "T3.2.8"
            and parent.get("status") == "PASS"
            and parent.get("all_required_gates_passed")
            and parent.get("implementation_hash_matches")
        ),
        "exact_four_cutoff_three_noise_on_off_matrix_is_present": lane_ids_match,
        "all_lanes_use_exactly_six_pauli_eigenstates": all(
            lane.get("state_labels") == list(STATE_LABELS) for lane in lanes.values()
        ) if lanes else False,
        "qec_on_off_pairs_are_matched_except_intervention": matched,
        "registered_cycle_is_ten_us_and_not_cross_platform_timing": all(
            abs(float(lane["config"]["cycle_duration_us"]) - 10.0) <= 1.0e-12
            and lane["config"]["scope"] == MODEL_SCOPE
            for lane in lanes.values()
        ) if lanes else False,
        "initial_code_subchannel_is_identity": initial_identity <= 2.0e-10,
        "initial_code_leakage_is_zero": initial_leakage <= 2.0e-10,
        "all_six_state_pair_sums_are_linear": all(
            float(point["pair_sum_linearity_residual"]) <= 2.0e-8
            for point in all_tomography
        ) if all_tomography else False,
        "all_reconstructed_choi_matrices_are_cp": all(
            float(point["minimum_choi_eigenvalue"]) >= -2.0e-8
            for point in all_tomography
        ) if all_tomography else False,
        "all_code_subchannels_are_trace_nonincreasing": all(
            min(point["tni_effect_eigenvalues"]) >= -2.0e-8
            and max(point["tni_effect_eigenvalues"]) <= 1.0 + 2.0e-8
            for point in all_tomography
        ) if all_tomography else False,
        "all_reconstructed_points_pass_physicality": all(
            point.get("passed_physicality") is True for point in all_tomography
        ) if all_tomography else False,
        "full_physical_density_diagnostics_pass": all(
            float(lane["maximum_physical_trace_error"]) <= 2.0e-10
            and float(lane["maximum_physical_hermiticity_error"]) <= 2.0e-10
            and float(lane["minimum_physical_eigenvalue"]) >= -2.0e-8
            for lane in lanes.values()
        ) if lanes else False,
        "leakage_is_retained_as_missing_trace": (
            any(float(point["mean_leakage"]) > 1.0e-6 for point in all_tomography)
            and event_ok
        ),
        "non_pauli_and_survival_diagnostics_are_not_hardcoded_zero": (
            any(float(point["off_diagonal_pauli_norm"]) > 1.0e-6 for point in all_tomography)
            and any(float(point["state_dependent_survival_norm"]) > 1.0e-8 for point in all_tomography)
        ),
        "ptm_uses_unnormalized_outputs_not_conditional_postselection": conditional_not_source,
        "pauli_lifetimes_preserve_raw_area_and_censoring": lifetime_units,
        "cycle_and_wallclock_lifetime_units_are_exactly_consistent": lifetime_units,
        "event_cost_is_protocol_native_and_qec_off_is_passive": event_ok,
        "cutoff_repeat_is_diagnostic_not_silent_convergence_claim": all(
            row.get("used_as_infinite_cutoff_convergence_claim") is False
            and np.isfinite(row.get("final_ptm_frobenius_difference", np.nan))
            for row in payload.get("cutoff_diagnostics", [])
        ) and len(payload.get("cutoff_diagnostics", [])) == 18,
        "terminal_36_to_40_cutoff_pair_meets_stability_tolerance": all(
            float(row["final_ptm_frobenius_difference"]) <= 0.03
            and float(row["final_mean_leakage_absolute_difference"]) <= 0.02
            for row in payload.get("cutoff_diagnostics", [])
            if row.get("terminal_registered_pair") is True
        ) and sum(
            row.get("terminal_registered_pair") is True
            for row in payload.get("cutoff_diagnostics", [])
        ) == 6,
        "heterogeneous_twirl_teacher_and_fault_results_are_excluded": (
            len(excluded) == 3 and all(row.get("status") == "EXCLUDED" for row in excluded)
        ),
        "no_target_hardware_or_experimental_tomography_claim": (
            payload.get("claim_boundary", {}).get("target_hardware_measured") is False
            and payload.get("claim_boundary", {}).get("experimental_tomography") is False
            and all(lane["event_accounting"]["target_hardware_measured"] == 0 for lane in lanes.values())
        ) if lanes else False,
        "no_desired_qec_performance_direction_is_required": all(
            comparison.get("performance_direction_required") is False
            for comparison in payload.get("matched_comparisons", [])
        ) and len(payload.get("matched_comparisons", [])) == 12,
        "all_numeric_channel_arrays_are_finite": all(
            np.all(np.isfinite(np.asarray(lane["survival"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(lane["leakage"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(lane["conditional_bloch_xyz"], dtype=np.float64)))
            and all(np.all(np.isfinite(np.asarray(point["ptm"], dtype=np.float64))) for point in lane["tomography"])
            for lane in lanes.values()
        ) if lanes else False,
        "raw_six_state_outputs_reproduce_tomography_and_lifetimes": (
            _raw_reconstruction_consistency(lanes) if lanes else False
        ),
        "cross_lane_comparison_and_cutoff_tables_recompute_exactly": (
            _deep_close(_matched_comparisons(lanes), payload.get("matched_comparisons", []))
            and _deep_close(_cutoff_diagnostics(lanes), payload.get("cutoff_diagnostics", []))
        ) if lanes else False,
    }


def validate_artifact_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    if payload.get("task_id") != TASK_ID or payload.get("contract_id") != CONTRACT_ID:
        raise ValueError("artifact task/contract identity mismatch")
    recomputed = _semantic_gates(payload)
    stored = payload.get("gates")
    if stored != recomputed:
        raise ValueError("stored gates do not match recomputed semantic gates")
    required = payload.get("required_gates")
    if required != list(recomputed):
        raise ValueError("required gate order/schema mismatch")
    expected_status = "PASS" if all(recomputed.values()) else "FAIL"
    if payload.get("status") != expected_status:
        raise ValueError("artifact status does not match gates")
    return recomputed


def run_benchmark(config: LogicalChannelBenchmarkConfig) -> dict[str, Any]:
    start = time.perf_counter()
    parent = _parent_audit()
    lanes: dict[str, Any] = {}
    for cutoff in config.cutoffs:
        for noise in NOISE_PROFILES:
            for mode in ("qec_off", "qec_on"):
                lane = _run_lane(config, cutoff=cutoff, noise=noise, mode=mode)
                lanes[lane["lane_id"]] = lane
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "status": "PENDING",
        "implementation_sha256": implementation_sha256(),
        "config": {**asdict(config), "cutoffs": list(config.cutoffs)},
        "noise_profiles_us": {
            key: {"cavity": value[0], "ancilla_t1": value[1], "ancilla_t2": value[2]}
            for key, value in NOISE_PROFILES.items()
        },
        "parent_audit": parent,
        "tomography_contract": {
            "state_labels": list(STATE_LABELS),
            "pauli_order": list(PAULI_LABELS),
            "ptm_reconstruction_source": "unnormalized_code_space_outputs",
            "channel_class": "completely_positive_trace_nonincreasing_code_subchannel",
            "leakage_definition": "one_minus_unnormalized_code_trace",
            "conditional_bloch_role": "diagnostic_only_not_channel_reconstruction",
            "lifetime_definition": "finite_horizon_raw_code_weighted_pauli_contrast_area_plus_optional_e_fold_crossing",
            "postselection": False,
        },
        "evidence_routing": {
            "used_parent": "T3.2.8 protocol-native finite-cutoff map/timing only",
            "excluded_heterogeneous_inputs": [
                {
                    "source": "physics/logical_channel.py",
                    "status": "EXCLUDED",
                    "reason": "parity-confusion Pauli twirl cannot supply coherent/non-Pauli/leakage tomography",
                },
                {
                    "source": "T4.4 teacher/student physical trajectories",
                    "status": "EXCLUDED",
                    "reason": "different learned-control horizon and no matched six-state channel output",
                },
                {
                    "source": "T5.2 causal fault artifacts",
                    "status": "EXCLUDED",
                    "reason": "component sensitivity estimands cannot be assembled into a logical channel",
                },
            ],
        },
        "lanes": lanes,
        "matched_comparisons": _matched_comparisons(lanes),
        "cutoff_diagnostics": _cutoff_diagnostics(lanes),
        "claim_boundary": {
            "allowed": "finite-cutoff matched-model CPTNI logical-channel reconstruction and simulation-derived QEC-on/off Pauli-signal comparison",
            "forbidden": [
                "experimental logical-channel tomography",
                "multilevel leakage dynamics",
                "physical-memory LER",
                "beyond-break-even claim",
                "device calibration",
                "target-board timing or FPGA measurement",
            ],
            "target_hardware_measured": False,
            "experimental_tomography": False,
            "performance_direction_preregistered": False,
        },
    }
    gates = _semantic_gates(payload)
    payload["gates"] = gates
    payload["required_gates"] = list(gates)
    payload["status"] = "PASS" if all(gates.values()) else "FAIL"
    payload["wall_time_seconds"] = time.perf_counter() - start
    validate_artifact_payload(payload)
    return payload


def _source_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "category": "contract",
            "task_id": TASK_ID,
            "contract_id": CONTRACT_ID,
            "implementation_sha256": payload["implementation_sha256"],
            "status": payload["status"],
        },
        {
            "category": "parent",
            **payload["parent_audit"],
        },
    ]
    for lane_id, lane in payload["lanes"].items():
        cycles = lane["cycles"]
        times = lane["time_us"]
        for cycle_index, (cycle, time_us) in enumerate(zip(cycles, times, strict=True)):
            for state_index, state_label in enumerate(STATE_LABELS):
                real = lane["projected_output_real"][cycle_index][state_index]
                imag = lane["projected_output_imag"][cycle_index][state_index]
                bloch = lane["conditional_bloch_xyz"][cycle_index][state_index]
                rows.append(
                    {
                        "category": "state_output",
                        "lane_id": lane_id,
                        "cutoff": lane["config"]["cutoff"],
                        "noise_profile": lane["noise_profile"],
                        "mode": lane["config"]["mode"],
                        "cycle": cycle,
                        "time_us": time_us,
                        "state_label": state_label,
                        "survival": lane["survival"][cycle_index][state_index],
                        "leakage": lane["leakage"][cycle_index][state_index],
                        "conditional_bloch_x": bloch[0],
                        "conditional_bloch_y": bloch[1],
                        "conditional_bloch_z": bloch[2],
                        "rho00_real": real[0][0],
                        "rho00_imag": imag[0][0],
                        "rho01_real": real[0][1],
                        "rho01_imag": imag[0][1],
                        "rho10_real": real[1][0],
                        "rho10_imag": imag[1][0],
                        "rho11_real": real[1][1],
                        "rho11_imag": imag[1][1],
                    }
                )
            tomography = lane["tomography"][cycle_index]
            for row_index, row_label in enumerate(PAULI_LABELS):
                for column_index, column_label in enumerate(PAULI_LABELS):
                    rows.append(
                        {
                            "category": "ptm",
                            "lane_id": lane_id,
                            "cutoff": lane["config"]["cutoff"],
                            "noise_profile": lane["noise_profile"],
                            "mode": lane["config"]["mode"],
                            "cycle": cycle,
                            "time_us": time_us,
                            "ptm_row": row_label,
                            "ptm_column": column_label,
                            "value": tomography["ptm"][row_index][column_index],
                        }
                    )
            rows.append(
                {
                    "category": "tomography_diagnostic",
                    "lane_id": lane_id,
                    "cutoff": lane["config"]["cutoff"],
                    "noise_profile": lane["noise_profile"],
                    "mode": lane["config"]["mode"],
                    "cycle": cycle,
                    "time_us": time_us,
                    **{
                        key: tomography[key]
                        for key in (
                            "pair_sum_linearity_residual",
                            "minimum_choi_eigenvalue",
                            "minimum_survival",
                            "maximum_survival",
                            "mean_leakage",
                            "survival_spread",
                            "off_diagonal_pauli_norm",
                            "coherent_rotation_norm",
                            "nonunital_code_flow_norm",
                            "state_dependent_survival_norm",
                            "passed_physicality",
                        )
                    },
                }
            )
        for axis, metric in lane["pauli_lifetimes"].items():
            rows.append(
                {
                    "category": "pauli_lifetime",
                    "lane_id": lane_id,
                    "cutoff": lane["config"]["cutoff"],
                    "noise_profile": lane["noise_profile"],
                    "mode": lane["config"]["mode"],
                    "axis": axis,
                    **metric,
                }
            )
    for comparison in payload["matched_comparisons"]:
        for axis, metrics in comparison["axes"].items():
            rows.append(
                {
                    "category": "matched_comparison",
                    "cutoff": comparison["cutoff"],
                    "noise_profile": comparison["noise_profile"],
                    "axis": axis,
                    **metrics,
                }
            )
    for diagnostic in payload["cutoff_diagnostics"]:
        rows.append({"category": "cutoff_diagnostic", **diagnostic})
    for gate, passed in payload["gates"].items():
        rows.append({"category": "gate", "gate": gate, "passed": passed})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    config: LogicalChannelBenchmarkConfig,
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = run_benchmark(config)
    rows = _source_rows(payload)
    _write_csv(source_data_path, rows)
    payload["source_data"] = {
        "path": source_data_path.as_posix(),
        "sha256": _sha256(source_data_path),
        "row_count": len(rows),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--cutoffs", type=int, nargs=4, default=(12, 24, 36, 40))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    payload = write_artifacts(
        LogicalChannelBenchmarkConfig(
            full_cycles=args.cycles,
            cutoffs=tuple(args.cutoffs),
            device=args.device,
        ),
        artifact_path=args.artifact,
        source_data_path=args.source_data,
    )
    print(
        json.dumps(
            {
                "task_id": payload["task_id"],
                "status": payload["status"],
                "gates": f"{sum(payload['gates'].values())}/{len(payload['gates'])}",
                "source_rows": payload["source_data"]["row_count"],
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "LogicalChannelBenchmarkConfig",
    "implementation_sha256",
    "run_benchmark",
    "validate_artifact_payload",
    "write_artifacts",
]
