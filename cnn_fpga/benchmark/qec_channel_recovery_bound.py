"""T5.3.5 QEC-matrix/Petz and small-cutoff SDP channel-recovery bound."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cnn_fpga.benchmark.logical_channel_fidelity import (
    implementation_sha256 as fidelity_implementation_sha256,
    validate_artifact_payload as validate_fidelity_payload,
)
from cnn_fpga.benchmark.logical_channel_reconstruction import (
    implementation_sha256 as channel_implementation_sha256,
    validate_artifact_payload as validate_channel_payload,
)
from cnn_fpga.benchmark.teacher_student_gain_retention import (
    implementation_sha256 as teacher_student_implementation_sha256,
)
from physics.channel_recovery_bound import (
    evaluate_encoded_channel_recovery,
    finite_cutoff_gkp_isometry,
    pure_loss_kraus,
)


TASK_ID = "T5.3.5"
CONTRACT_ID = "T535-QEC-MATRIX-PETZ-SDP-RECOVERY-BOUND-V1"
DEFAULT_ARTIFACT = Path("docs/t5_3_5_qec_channel_recovery_bound.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_3_5_qec_channel_recovery_bound_source_data.csv")
PARENT_PATHS = {
    "T5.3.1": Path("docs/t5_3_1_logical_channel_reconstruction.json"),
    "T5.3.2": Path("docs/t5_3_2_logical_channel_fidelity.json"),
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention.json"),
}
NOISE_PROFILES: Mapping[str, tuple[float, float, float]] = {
    "high": (245.0, 50.0, 60.0),
    "medium": (490.0, 100.0, 120.0),
    "low": (610.0, 280.0, 238.0),
}


@dataclass(frozen=True)
class RecoveryBoundBenchmarkConfig:
    small_sdp_cutoffs: tuple[int, ...] = (4, 6, 8, 10, 12)
    extended_cutoffs: tuple[int, ...] = (12, 24, 36, 40, 48)
    energy_cutoffs: tuple[int, ...] = (24, 36, 48)
    energy_projector_deltas: tuple[float, ...] = (0.44, 0.34, 0.28)
    projector_delta: float = 0.34
    cycle_duration_us: float = 10.0
    grid_points: int = 8193
    solver: str = "CLARABEL"
    solver_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in ("small_sdp_cutoffs", "extended_cutoffs", "energy_cutoffs"):
            values = tuple(int(value) for value in getattr(self, name))
            if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
                raise ValueError(f"{name} must be a nonempty strictly increasing tuple")
            if any(not 4 <= value <= 48 for value in values):
                raise ValueError(f"{name} values must lie in [4,48]")
            object.__setattr__(self, name, values)
        if self.small_sdp_cutoffs != (4, 6, 8, 10, 12):
            raise ValueError("small SDP validation cutoffs are frozen to 4,6,8,10,12")
        if self.extended_cutoffs != (12, 24, 36, 40, 48):
            raise ValueError("extended cutoffs are frozen to 12,24,36,40,48")
        if self.energy_cutoffs != (24, 36, 48):
            raise ValueError("energy cutoffs are frozen to 24,36,48")
        deltas = tuple(float(value) for value in self.energy_projector_deltas)
        if deltas != (0.44, 0.34, 0.28):
            raise ValueError("energy projector deltas are frozen to 0.44,0.34,0.28")
        object.__setattr__(self, "energy_projector_deltas", deltas)
        if float(self.projector_delta) != 0.34:
            raise ValueError("matched sBs comparison requires projector_delta=0.34")
        if float(self.cycle_duration_us) != 10.0:
            raise ValueError("matched channel-recovery interval is frozen to 10 us")
        if int(self.grid_points) != 8193:
            raise ValueError("formal finite-Fock GKP projection requires 8193 grid points")
        if str(self.solver).upper() != "CLARABEL":
            raise ValueError("small-cutoff certificate requires CLARABEL")
        if not np.isfinite(self.solver_tolerance) or not 0.0 < self.solver_tolerance <= 1.0e-7:
            raise ValueError("solver_tolerance must lie in (0,1e-7]")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "channel_recovery_bound.py",
        Path(__file__).resolve().parents[2] / "physics" / "fock_density_model.py",
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_parents() -> dict[str, dict[str, Any]]:
    parents = {
        task: json.loads(path.read_text(encoding="utf-8"))
        for task, path in PARENT_PATHS.items()
    }
    validate_channel_payload(parents["T5.3.1"])
    validate_fidelity_payload(parents["T5.3.2"])
    teacher = parents["T4.4.4"]
    if teacher.get("task_id") != "T4.4.4" or teacher.get("status") != "PASS":
        raise ValueError("T4.4.4 teacher/student parent is not PASS")
    if not teacher.get("gates") or not all(teacher["gates"].values()):
        raise ValueError("T4.4.4 teacher/student parent gates are not all true")
    if teacher.get("implementation_sha256") != teacher_student_implementation_sha256():
        raise ValueError("T4.4.4 teacher/student implementation hash is stale")
    return parents


def _live_parent_hash(task: str) -> str:
    return {
        "T5.3.1": channel_implementation_sha256(),
        "T5.3.2": fidelity_implementation_sha256(),
        "T4.4.4": teacher_student_implementation_sha256(),
    }[task]


def _parent_audits(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    audits = {}
    for task, path in PARENT_PATHS.items():
        payload = parents[task]
        stored = payload.get("implementation_sha256")
        live = _live_parent_hash(task)
        gates = payload.get("gates", {})
        audits[task] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "task_id": payload.get("task_id"),
            "status": payload.get("status"),
            "all_gates_passed": bool(gates) and all(gates.values()),
            "stored_implementation_sha256": stored,
            "live_implementation_sha256": live,
            "implementation_hash_matches": stored == live,
        }
    return audits


def _row_id(cutoff: int, delta: float, noise: str) -> str:
    return f"cutoff{cutoff}:delta{delta:.2f}:{noise}"


def _evaluate_row(
    config: RecoveryBoundBenchmarkConfig,
    isometry: np.ndarray,
    *,
    cutoff: int,
    delta: float,
    noise: str,
    solve_sdp: bool,
) -> dict[str, Any]:
    cavity_lifetime, ancilla_t1, ancilla_t2 = NOISE_PROFILES[noise]
    kraus = pure_loss_kraus(
        cutoff,
        duration_us=config.cycle_duration_us,
        cavity_lifetime_us=cavity_lifetime,
    )
    result = evaluate_encoded_channel_recovery(
        isometry,
        kraus,
        solve_sdp=solve_sdp,
        solver=config.solver,
        solver_tolerance=config.solver_tolerance,
    ).to_dict()
    return {
        "row_id": _row_id(cutoff, delta, noise),
        "cutoff": cutoff,
        "projector_delta": delta,
        "noise_profile": noise,
        "cavity_lifetime_us": cavity_lifetime,
        "ancilla_t1_us_comparison_only": ancilla_t1,
        "ancilla_t2_us_comparison_only": ancilla_t2,
        "channel_duration_us": config.cycle_duration_us,
        "pure_loss_transmissivity": exp(-config.cycle_duration_us / cavity_lifetime),
        "pre_recovery_noise_channel": "exact_finite_cutoff_cavity_pure_loss",
        "ancilla_noise_in_bound": False,
        "arbitrary_terminal_recovery_allowed": True,
        **result,
    }


def _build_bound_rows(config: RecoveryBoundBenchmarkConfig) -> dict[str, list[dict[str, Any]]]:
    isometries: dict[tuple[int, float], np.ndarray] = {}

    def get_isometry(cutoff: int, delta: float) -> np.ndarray:
        key = (cutoff, delta)
        if key not in isometries:
            isometries[key] = finite_cutoff_gkp_isometry(
                cutoff,
                delta,
                grid_points=config.grid_points,
            )
        return isometries[key]

    small = [
        _evaluate_row(
            config,
            get_isometry(cutoff, config.projector_delta),
            cutoff=cutoff,
            delta=config.projector_delta,
            noise=noise,
            solve_sdp=True,
        )
        for cutoff in config.small_sdp_cutoffs
        for noise in NOISE_PROFILES
    ]
    extended = [
        _evaluate_row(
            config,
            get_isometry(cutoff, config.projector_delta),
            cutoff=cutoff,
            delta=config.projector_delta,
            noise=noise,
            solve_sdp=False,
        )
        for cutoff in config.extended_cutoffs
        for noise in NOISE_PROFILES
    ]
    energy = [
        _evaluate_row(
            config,
            get_isometry(cutoff, delta),
            cutoff=cutoff,
            delta=delta,
            noise=noise,
            solve_sdp=False,
        )
        for delta in config.energy_projector_deltas
        for cutoff in config.energy_cutoffs
        for noise in NOISE_PROFILES
    ]
    return {
        "small_sdp_validation": small,
        "extended_cutoff_scan": extended,
        "energy_extension_scan": energy,
    }


def _cutoff_diagnostics(rows: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    extended = {
        (row["cutoff"], row["noise_profile"]): row
        for row in rows["extended_cutoff_scan"]
    }
    for noise in NOISE_PROFILES:
        for lower, higher in zip((12, 24, 36, 40), (24, 36, 40, 48), strict=True):
            low = extended[(lower, noise)]
            high = extended[(higher, noise)]
            diagnostics.append(
                {
                    "scan": "matched_delta_cutoff",
                    "noise_profile": noise,
                    "projector_delta": 0.34,
                    "lower_cutoff": lower,
                    "higher_cutoff": higher,
                    "terminal_pair": (lower, higher) == (40, 48),
                    "petz_fidelity_difference": high["petz"]["petz_fidelity"]
                    - low["petz"]["petz_fidelity"],
                    "absolute_petz_fidelity_difference": abs(
                        high["petz"]["petz_fidelity"]
                        - low["petz"]["petz_fidelity"]
                    ),
                    "mean_photon_number_difference": high["mean_code_photon_number"]
                    - low["mean_code_photon_number"],
                    "infinite_cutoff_convergence_claimed": False,
                }
            )
    energy = {
        (row["projector_delta"], row["cutoff"], row["noise_profile"]): row
        for row in rows["energy_extension_scan"]
    }
    for delta in (0.44, 0.34, 0.28):
        for noise in NOISE_PROFILES:
            low = energy[(delta, 36, noise)]
            high = energy[(delta, 48, noise)]
            diagnostics.append(
                {
                    "scan": "energy_terminal_cutoff",
                    "noise_profile": noise,
                    "projector_delta": delta,
                    "lower_cutoff": 36,
                    "higher_cutoff": 48,
                    "terminal_pair": True,
                    "petz_fidelity_difference": high["petz"]["petz_fidelity"]
                    - low["petz"]["petz_fidelity"],
                    "absolute_petz_fidelity_difference": abs(
                        high["petz"]["petz_fidelity"]
                        - low["petz"]["petz_fidelity"]
                    ),
                    "mean_photon_number_difference": high["mean_code_photon_number"]
                    - low["mean_code_photon_number"],
                    "infinite_cutoff_convergence_claimed": False,
                }
            )
    return diagnostics


def _actual_sbs_gap_rows(
    parents: Mapping[str, Mapping[str, Any]],
    rows: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    fidelity = parents["T5.3.2"]
    extended = {
        (row["cutoff"], row["noise_profile"]): row
        for row in rows["extended_cutoff_scan"]
    }
    small = {
        (row["cutoff"], row["noise_profile"]): row
        for row in rows["small_sdp_validation"]
    }
    comparisons = []
    for cutoff in (12, 24, 36, 40):
        for noise in NOISE_PROFILES:
            parent_lane_id = f"cutoff{cutoff}:{noise}:qec_on"
            point = fidelity["lanes"][parent_lane_id]["cycle_metrics"][1]
            raw_fidelity = float(point["entanglement_fidelity"])
            survival = float(point["mean_code_survival"])
            mean_leakage = 1.0 - survival
            completion = mean_leakage / 4.0
            completed = raw_fidelity + completion
            bound = extended[(cutoff, noise)]
            petz_lower = float(bound["petz"]["theorem_optimal_lower"])
            theorem_upper = float(bound["petz"]["theorem_optimal_upper"])
            if cutoff in {key[0] for key in small}:
                sdp = small[(cutoff, noise)]["sdp"]
                certified_lower = float(sdp["intersection_certified_lower"])
                certified_upper = float(sdp["intersection_certified_upper"])
                interval_source = "petz_theorem_intersect_repaired_primal_dual_sdp"
            else:
                certified_lower = petz_lower
                certified_upper = theorem_upper
                interval_source = "petz_theorem_only_no_large_cutoff_sdp"
            comparisons.append(
                {
                    "row_id": f"cutoff{cutoff}:{noise}:actual_sbs_gap",
                    "cutoff": cutoff,
                    "noise_profile": noise,
                    "parent_lane_id": parent_lane_id,
                    "cycle_index": 1,
                    "elapsed_time_us": 10.0,
                    "actual_sbs_cptni_entanglement_fidelity": raw_fidelity,
                    "actual_sbs_mean_code_survival": survival,
                    "actual_sbs_mean_code_leakage": mean_leakage,
                    "maximally_mixed_leakage_completion_contribution": completion,
                    "actual_sbs_completed_entanglement_fidelity": completed,
                    "completion_rule": (
                        "terminal decode Vdagger rho V plus mean-leakage replacement by I_L/2; "
                        "adds mean_leakage/d_L^2 to F_e"
                    ),
                    "petz_bound_lower": petz_lower,
                    "petz_theorem_upper": theorem_upper,
                    "comparison_bound_lower": certified_lower,
                    "comparison_bound_upper": certified_upper,
                    "comparison_bound_interval_source": interval_source,
                    "bound_minus_actual_sbs_gap_lower": certified_lower - completed,
                    "bound_minus_actual_sbs_gap_upper": certified_upper - completed,
                    "comparison_status": "SCHEDULE_MISMATCHED_DIAGNOSTIC_ONLY",
                    "schedule_mismatch": (
                        "bound permits one arbitrary terminal recovery after 10 us cavity pure loss; "
                        "actual nominal sBs interleaves two gate-reset rounds with cavity and ancilla noise"
                    ),
                    "certified_ordering_claimed": False,
                    "deployable_decoder_gap_claimed": False,
                }
            )
    return comparisons


def _teacher_student_gap_rows(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = parent["config"]
    rows = []
    for cutoff in (config["cutoff"], config["confirmation_cutoff"]):
        for role in ("teacher", "student"):
            rows.append(
                {
                    "row_id": f"cutoff{cutoff}:{role}:recovery_bound_gap",
                    "cutoff": cutoff,
                    "controller_role": role,
                    "source_task": "T4.4.4",
                    "source_horizon_cycles": config["full_cycles"],
                    "source_metric": "trajectory selection score, fidelity lifetime and logical-Z lifetime",
                    "bound_horizon_cycles": 1,
                    "bound_metric": "encoding-pure-loss arbitrary-terminal-recovery entanglement fidelity",
                    "recovery_bound_gap": None,
                    "status": "INCOMPARABLE",
                    "reason": (
                        "T4.4.4 is a ten-cycle two-level history-conditioned controller lane without "
                        "matched six-state channel Choi data; lifetime cannot be subtracted from one-cycle F_e"
                    ),
                    "heterogeneous_metric_subtraction_performed": False,
                }
            )
    return rows


def _derive(
    config: RecoveryBoundBenchmarkConfig,
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = _build_bound_rows(config)
    diagnostics = _cutoff_diagnostics(rows)
    actual = _actual_sbs_gap_rows(parents, rows)
    incomparable = _teacher_student_gap_rows(parents["T4.4.4"])
    return {
        **rows,
        "cutoff_diagnostics": diagnostics,
        "actual_sbs_gap_diagnostics": actual,
        "teacher_student_gap_audit": incomparable,
    }


def _semantic_gates(
    payload: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    audits = payload.get("parent_audits", {})
    small = payload.get("small_sdp_validation", [])
    extended = payload.get("extended_cutoff_scan", [])
    energy = payload.get("energy_extension_scan", [])
    cutoff = payload.get("cutoff_diagnostics", [])
    actual = payload.get("actual_sbs_gap_diagnostics", [])
    teacher = payload.get("teacher_student_gap_audit", [])
    claim = payload.get("claim_boundary", {})
    verdict = payload.get("verdict", {})
    expected_parent_tasks = set(PARENT_PATHS)
    parent_gate = lambda task: bool(
        audits.get(task, {}).get("path") == PARENT_PATHS[task].as_posix()
        and audits.get(task, {}).get("sha256") == _sha256(PARENT_PATHS[task])
        and audits.get(task, {}).get("task_id") == task
        and audits.get(task, {}).get("status") == "PASS"
        and audits.get(task, {}).get("all_gates_passed")
        and audits.get(task, {}).get("implementation_hash_matches")
    )
    small_keys = {(row.get("cutoff"), row.get("noise_profile")) for row in small}
    extended_keys = {(row.get("cutoff"), row.get("noise_profile")) for row in extended}
    energy_keys = {
        (row.get("projector_delta"), row.get("cutoff"), row.get("noise_profile"))
        for row in energy
    }
    return {
        "all_three_parent_artifacts_and_implementations_are_live": (
            set(audits) == expected_parent_tasks
            and all(parent_gate(task) for task in expected_parent_tasks)
        ),
        "formal_config_is_frozen": json.loads(json.dumps(payload.get("config")))
        == json.loads(json.dumps(asdict(RecoveryBoundBenchmarkConfig()))),
        "fifteen_small_cutoff_noise_sdp_lanes_are_complete": (
            len(small) == 15
            and small_keys
            == {(value, noise) for value in (4, 6, 8, 10, 12) for noise in NOISE_PROFILES}
        ),
        "all_encodings_and_encoded_channels_are_physical": all(
            row["encoding_orthonormality_residual"] <= 2.0e-10
            and row["encoded_channel_tp_residual"] <= 2.0e-10
            for row in small + extended + energy
        ) if small and extended and energy else False,
        "all_qec_matrices_are_psd_trace_consistent": all(
            row["petz"]["qec_matrix_minimum_eigenvalue"] >= -2.0e-8
            and row["petz"]["qec_matrix_trace_residual"] <= 2.0e-9
            and row["petz"]["qec_matrix_hermiticity_residual"] <= 2.0e-9
            for row in small + extended + energy
        ) if small and extended and energy else False,
        "qec_matrix_and_direct_petz_fidelities_agree_with_conditioning_audit": all(
            row["petz"]["qec_vs_direct_petz_residual"] <= 2.0e-7
            and row["petz_recovery"]["petz_support_tp_residual"] <= 2.0e-4
            and row["petz_recovery"]["encoded_output_outside_support_residual"] <= 2.0e-6
            for row in small + extended + energy
        ) if small and extended and energy else False,
        "petz_theorem_intervals_are_ordered_and_bounded": all(
            0.0 <= row["petz"]["theorem_optimal_lower"]
            <= row["petz"]["theorem_optimal_upper"] <= 1.0
            and row["petz"]["theorem_optimal_lower"] == row["petz"]["petz_fidelity"]
            for row in small + extended + energy
        ) if small and extended and energy else False,
        "all_small_cutoff_solver_statuses_are_accepted_with_repaired_certificates": all(
            row.get("sdp", {}).get("primal_status") in {"optimal", "optimal_inaccurate"}
            and row.get("sdp", {}).get("dual_status") in {"optimal", "optimal_inaccurate"}
            and row.get("sdp", {}).get("solver") == "CLARABEL"
            for row in small
        ) if small else False,
        "all_repaired_primal_choi_certificates_are_cptp": all(
            row["sdp"]["repaired_primal_minimum_eigenvalue"] >= -2.0e-8
            and row["sdp"]["repaired_primal_tp_residual"] <= 2.0e-8
            for row in small
        ) if small else False,
        "all_shifted_dual_certificates_are_feasible": all(
            row["sdp"]["repaired_dual_minimum_slack_eigenvalue"] >= 0.9e-10
            and row["sdp"]["repaired_dual_fidelity_upper"]
            >= row["sdp"]["repaired_primal_fidelity_lower"] - 2.0e-7
            for row in small
        ) if small else False,
        "all_sdp_and_petz_theorem_intersections_are_nonempty": all(
            row["petz"]["theorem_optimal_lower"]
            <= row["sdp"]["intersection_certified_upper"] + 2.0e-7
            and row["sdp"]["intersection_certified_lower"]
            <= row["petz"]["theorem_optimal_upper"] + 2.0e-7
            and row["sdp"]["intersection_width"] >= -2.0e-7
            for row in small
        ) if small else False,
        "small_cutoff_sdp_certificates_are_numerically_tight": all(
            row["sdp"]["repaired_certificate_width"] <= 2.0e-6
            and abs(row["sdp"]["raw_solver_duality_gap"]) <= 2.0e-6
            for row in small
        ) if small else False,
        "fifteen_extended_cutoff_noise_bounds_reach_cutoff_48": (
            len(extended) == 15
            and extended_keys
            == {(value, noise) for value in (12, 24, 36, 40, 48) for noise in NOISE_PROFILES}
        ),
        "twenty_seven_energy_extension_rows_are_complete": (
            len(energy) == 27
            and energy_keys
            == {
                (delta, value, noise)
                for delta in (0.44, 0.34, 0.28)
                for value in (24, 36, 48)
                for noise in NOISE_PROFILES
            }
        ),
        "smaller_delta_increases_registered_mean_code_energy_at_cutoff48": all(
            next(
                row["mean_code_photon_number"]
                for row in energy
                if row["projector_delta"] == 0.28
                and row["cutoff"] == 48
                and row["noise_profile"] == noise
            )
            > next(
                row["mean_code_photon_number"]
                for row in energy
                if row["projector_delta"] == 0.44
                and row["cutoff"] == 48
                and row["noise_profile"] == noise
            )
            for noise in NOISE_PROFILES
        ) if len(energy) == 27 else False,
        "all_cutoff_diagnostics_retain_finite_cutoff_boundary": (
            len(cutoff) == 21
            and all(row["infinite_cutoff_convergence_claimed"] is False for row in cutoff)
        ),
        "twelve_actual_sbs_rows_are_parent_bound_and_completed": (
            len(actual) == 12
            and all(
                row["actual_sbs_cptni_entanglement_fidelity"]
                == parents["T5.3.2"]["lanes"][row["parent_lane_id"]]["cycle_metrics"][1][
                    "entanglement_fidelity"
                ]
                and row["actual_sbs_mean_code_survival"]
                == parents["T5.3.2"]["lanes"][row["parent_lane_id"]]["cycle_metrics"][1][
                    "mean_code_survival"
                ]
                and abs(
                    row["actual_sbs_completed_entanglement_fidelity"]
                    - (
                        row["actual_sbs_cptni_entanglement_fidelity"]
                        + (1.0 - row["actual_sbs_mean_code_survival"]) / 4.0
                    )
                )
                <= 2.0e-12
                for row in actual
            )
        ),
        "actual_sbs_gaps_are_explicitly_schedule_mismatched_diagnostics": all(
            row["comparison_status"] == "SCHEDULE_MISMATCHED_DIAGNOSTIC_ONLY"
            and row["certified_ordering_claimed"] is False
            and row["deployable_decoder_gap_claimed"] is False
            and row["bound_minus_actual_sbs_gap_lower"]
            == row["comparison_bound_lower"]
            - row["actual_sbs_completed_entanglement_fidelity"]
            and row["bound_minus_actual_sbs_gap_upper"]
            == row["comparison_bound_upper"]
            - row["actual_sbs_completed_entanglement_fidelity"]
            for row in actual
        ) if actual else False,
        "teacher_student_gaps_are_null_and_incomparable": (
            len(teacher) == 4
            and all(
                row["recovery_bound_gap"] is None
                and row["status"] == "INCOMPARABLE"
                and row["heterogeneous_metric_subtraction_performed"] is False
                for row in teacher
            )
        ),
        "verdict_separates_bound_sdp_actual_and_heterogeneous_lanes": bool(
            verdict.get("petz_channel_recovery_bound") == "ESTABLISHED_FINITE_CUTOFF_PURE_LOSS"
            and verdict.get("small_cutoff_sdp_validation") == "PASS"
            and verdict.get("actual_sbs_gap") == "DIAGNOSTIC_ONLY_SCHEDULE_MISMATCHED"
            and verdict.get("teacher_student_gap") == "INCOMPARABLE_NULL"
            and verdict.get("large_cutoff_sdp_optimum") == "NOT_COMPUTED"
        ),
        "claim_boundary_remains_arbitrary_recovery_not_deployable_decoder": bool(
            claim.get("arbitrary_recovery_assumption") is True
            and claim.get("deployable_decoder_or_controller") is False
            and claim.get("hardware_result") is False
            and claim.get("infinite_cutoff_or_energy_convergence") is False
            and claim.get("actual_sbs_certified_below_bound") is False
            and claim.get("teacher_student_numeric_gap") is False
        ),
    }


def validate_artifact_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    if payload.get("task_id") != TASK_ID or payload.get("contract_id") != CONTRACT_ID:
        raise ValueError("artifact task/contract identity mismatch")
    if payload.get("implementation_sha256") != implementation_sha256():
        raise ValueError("artifact implementation hash is stale")
    parents = _load_parents()
    gates = _semantic_gates(payload, parents)
    if payload.get("gates") != gates:
        raise ValueError("stored gates do not match recomputed semantic gates")
    if payload.get("required_gates") != list(gates):
        raise ValueError("required gate order/schema mismatch")
    expected = "PASS" if all(gates.values()) else "FAIL"
    if payload.get("status") != expected:
        raise ValueError("artifact status does not match gates")
    return gates


def run_report(config: RecoveryBoundBenchmarkConfig | None = None) -> dict[str, Any]:
    actual_config = RecoveryBoundBenchmarkConfig() if config is None else config
    if not isinstance(actual_config, RecoveryBoundBenchmarkConfig):
        raise TypeError("config must be a RecoveryBoundBenchmarkConfig")
    parents = _load_parents()
    derived = _derive(actual_config, parents)
    small = derived["small_sdp_validation"]
    cutoff = derived["cutoff_diagnostics"]
    actual = derived["actual_sbs_gap_diagnostics"]
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "status": "PENDING",
        "implementation_sha256": implementation_sha256(),
        "config": asdict(actual_config),
        "parent_audits": _parent_audits(parents),
        "metric_contract": {
            "encoding": "orthonormal finite-cutoff square approximate-GKP qubit",
            "noise": "exact finite-cutoff cavity pure loss for one 10 us interval",
            "qec_matrix_order": "logical-major then Kraus-minor",
            "primary_metric": "channel/entanglement fidelity to logical identity",
            "petz_formula": "||Tr_L sqrt(M)||_F^2 / d_L^2",
            "theorem_interval": "F_Petz <= F_opt <= (1+F_Petz)/2",
            "sdp_primal": "CPTP recovery Choi feasible lower certificate",
            "sdp_dual": "shifted dual-feasible upper certificate",
            "recovery_scope": "arbitrary terminal CPTP recovery after encoding and noise",
            "deployable_decoder_or_controller": False,
        },
        **derived,
        "verdict": {
            "petz_channel_recovery_bound": "ESTABLISHED_FINITE_CUTOFF_PURE_LOSS",
            "small_cutoff_sdp_validation": "PASS",
            "small_cutoff_sdp_lanes": len(small),
            "maximum_repaired_sdp_certificate_width": max(
                row["sdp"]["repaired_certificate_width"] for row in small
            ),
            "maximum_petz_below_sdp_optimum": max(
                row["sdp"]["intersection_certified_lower"]
                - row["petz"]["petz_fidelity"]
                for row in small
            ),
            "large_cutoff_sdp_optimum": "NOT_COMPUTED",
            "maximum_terminal_cutoff_petz_difference": max(
                row["absolute_petz_fidelity_difference"]
                for row in cutoff
                if row["terminal_pair"]
            ),
            "infinite_cutoff_or_energy_convergence": "NOT_ESTABLISHED",
            "actual_sbs_gap": "DIAGNOSTIC_ONLY_SCHEDULE_MISMATCHED",
            "actual_sbs_gap_interval_minimum": min(
                row["bound_minus_actual_sbs_gap_lower"] for row in actual
            ),
            "actual_sbs_gap_interval_maximum": max(
                row["bound_minus_actual_sbs_gap_upper"] for row in actual
            ),
            "teacher_student_gap": "INCOMPARABLE_NULL",
            "paper_or_experimental_recovery_bound": "NOT_ESTABLISHED",
        },
        "claim_boundary": {
            "allowed": (
                "finite-cutoff pure-loss encoding--noise Petz/theorem interval, small-cutoff "
                "primal/dual SDP certificate, cutoff/energy sensitivity, and explicitly "
                "schedule-mismatched sBs diagnostic"
            ),
            "forbidden": [
                "Petz bound as a per-shot MAP decoder or deployable controller",
                "arbitrary recovery as an implementable sBs pulse sequence",
                "actual interleaved sBs as a certified feasible point of the terminal-recovery SDP",
                "teacher/student lifetime minus one-cycle channel fidelity",
                "cutoff 48 as infinite-dimensional convergence",
                "ancilla/device noise, hardware timing, experimental fidelity or LER claim",
            ],
            "arbitrary_recovery_assumption": True,
            "deployable_decoder_or_controller": False,
            "hardware_result": False,
            "infinite_cutoff_or_energy_convergence": False,
            "actual_sbs_certified_below_bound": False,
            "teacher_student_numeric_gap": False,
        },
    }
    gates = _semantic_gates(payload, parents)
    payload["gates"] = gates
    payload["required_gates"] = list(gates)
    payload["status"] = "PASS" if all(gates.values()) else "FAIL"
    validate_artifact_payload(payload)
    return payload


def _source_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "category": "contract",
            "task_id": TASK_ID,
            "contract_id": CONTRACT_ID,
            "status": payload["status"],
            "implementation_sha256": payload["implementation_sha256"],
        }
    ]
    rows.extend(
        {"category": "parent", "parent_task": task, **audit}
        for task, audit in payload["parent_audits"].items()
    )
    for category in ("small_sdp_validation", "extended_cutoff_scan", "energy_extension_scan"):
        for row in payload[category]:
            flat = {
                key: value
                for key, value in row.items()
                if key not in {"petz", "petz_recovery", "sdp"}
            }
            flat.update({f"petz_{key}": value for key, value in row["petz"].items()})
            flat.update(
                {f"petz_recovery_{key}": value for key, value in row["petz_recovery"].items()}
            )
            if row["sdp"] is not None:
                flat.update({f"sdp_{key}": value for key, value in row["sdp"].items()})
            rows.append({"category": category, **flat})
    rows.extend(
        {"category": "cutoff_diagnostic", **row}
        for row in payload["cutoff_diagnostics"]
    )
    rows.extend(
        {"category": "actual_sbs_gap", **row}
        for row in payload["actual_sbs_gap_diagnostics"]
    )
    rows.extend(
        {"category": "teacher_student_gap", **row}
        for row in payload["teacher_student_gap_audit"]
    )
    rows.extend(
        {"category": "gate", "gate": gate, "passed": passed}
        for gate, passed in payload["gates"].items()
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = run_report()
    rows = _source_rows(payload)
    _write_csv(source_data_path, rows)
    payload["source_data"] = {
        "path": source_data_path.as_posix(),
        "sha256": _sha256(source_data_path),
        "row_count": len(rows),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args()
    payload = write_artifacts(
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
                "verdict": payload["verdict"],
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
    "RecoveryBoundBenchmarkConfig",
    "implementation_sha256",
    "run_report",
    "validate_artifact_payload",
    "write_artifacts",
]
