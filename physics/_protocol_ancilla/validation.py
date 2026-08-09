"""Validation, secondary-protocol registry, and JSON reporting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..constants import LATTICE_CONST
from ..sbs_observation_reset import (
    SBSObservationResetModel,
    make_persistent_leakage_model,
)
from .common import _seed
from .fault_overlay import SBSAncillaFaultOverlay, SBSFaultOverlayConfig
from .sharpen_trim import SharpenTrimAncillaConfig, SharpenTrimAncillaModel

@dataclass(frozen=True)
class SecondaryProtocolNoiseSpec:
    protocol_id: str
    role: str
    executable: bool
    allowed_scan_parameters: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    primary_source_required: bool = True


SECONDARY_PROTOCOL_NOISE_REGISTRY: Mapping[str, SecondaryProtocolNoiseSpec] = MappingProxyType(
    {
        "steane": SecondaryProtocolNoiseSpec(
            protocol_id="PROTO-STEANE-THEORY-ONLY",
            role="theoretical_comparator",
            executable=False,
            allowed_scan_parameters=("data_variance", "ancilla_variance"),
            forbidden_claims=("SBS main ranking", "hardware implementation", "device timing"),
        ),
        "knill_qunaught": SecondaryProtocolNoiseSpec(
            protocol_id="PROTO-KNILL-QUNAUGHT-SECONDARY",
            role="secondary_noise_shaping_comparator",
            executable=False,
            allowed_scan_parameters=("resource_squeezing", "homodyne_variance"),
            forbidden_claims=("SBS main ranking", "FPGA physical control", "verified 1e-8"),
        ),
        "p_steane": SecondaryProtocolNoiseSpec(
            protocol_id="PROTO-ME-PSTEANE-SECONDARY",
            role="secondary_tunable_preprocessing_comparator",
            executable=False,
            allowed_scan_parameters=("a", "b", "data_to_ancilla_noise_ratio"),
            forbidden_claims=("FPGA implements squeezing", "project-verified optimum", "device timing"),
        ),
    }
)


def secondary_protocol_noise_specs() -> tuple[SecondaryProtocolNoiseSpec, ...]:
    return tuple(SECONDARY_PROTOCOL_NOISE_REGISTRY.values())


@dataclass(frozen=True)
class ProtocolAncillaValidationResult:
    samples: int
    seed: int
    sbs_expected_big_cd_bit_rate: float
    sbs_observed_big_cd_bit_rate: float
    sbs_expected_logical_backaction_rate: float
    sbs_observed_logical_backaction_rate: float
    sbs_phase_outcome_toggle_rate: float
    sharpen_expected_feedback_mismatch_rate: float
    sharpen_observed_feedback_mismatch_rate: float
    sharpen_bit_middle_window_rate: float
    sharpen_bit_logical_backaction_rate: float
    checks: Mapping[str, bool]
    evidence_scope: str = "protocol_native_ancilla_error_effective_validation_not_device_calibrated"

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "sbs": {
                "expected_big_cd_bit_rate": self.sbs_expected_big_cd_bit_rate,
                "observed_big_cd_bit_rate": self.sbs_observed_big_cd_bit_rate,
                "expected_logical_backaction_rate": self.sbs_expected_logical_backaction_rate,
                "observed_logical_backaction_rate": self.sbs_observed_logical_backaction_rate,
                "phase_outcome_toggle_rate": self.sbs_phase_outcome_toggle_rate,
            },
            "sharpen_trim": {
                "expected_feedback_mismatch_rate": self.sharpen_expected_feedback_mismatch_rate,
                "observed_feedback_mismatch_rate": self.sharpen_observed_feedback_mismatch_rate,
                "bit_middle_window_rate": self.sharpen_bit_middle_window_rate,
                "bit_logical_backaction_rate": self.sharpen_bit_logical_backaction_rate,
            },
            "checks": {name: bool(value) for name, value in self.checks.items()},
            "secondary_protocols": [
                {
                    "protocol_id": spec.protocol_id,
                    "role": spec.role,
                    "executable": spec.executable,
                    "allowed_scan_parameters": list(spec.allowed_scan_parameters),
                    "forbidden_claims": list(spec.forbidden_claims),
                }
                for spec in secondary_protocol_noise_specs()
            ],
            "evidence_scope": self.evidence_scope,
            "claim_boundary": {
                "allowed": "protocol-native effective ancilla/readout/reset fault flow",
                "forbidden": "device-calibrated fault rates, master-equation fidelity, or secondary-protocol hardware execution",
            },
        }


def _perfect_sbs_base_model() -> SBSObservationResetModel:
    return make_persistent_leakage_model(
        readout_confusion=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=1.0,
        counter_max=2**31 - 1,
        readout_provenance="T2.2.2 validation perfect classifier assumption",
        parameter_provenance="T2.2.2 validation no-leakage perfect-reset assumption",
    )


def run_protocol_ancilla_validation(
    *,
    samples: int = 60_000,
    seed: int = 2026071422,
) -> ProtocolAncillaValidationResult:
    if not isinstance(samples, int) or samples < 10_000:
        raise ValueError("samples must be an integer >= 10000")
    normalized_seed = _seed(seed)
    bit_rate = 0.02
    phase_rate = 0.03
    logical_given_bit = 0.5
    sbs = SBSAncillaFaultOverlay(
        _perfect_sbs_base_model(),
        SBSFaultOverlayConfig(
            bit_flip_probabilities=((0.0, bit_rate, 0.0), (0.0, 0.0, 0.0)),
            phase_flip_probabilities=((0.0, 0.0, 0.0), (0.0, phase_rate, 0.0)),
            logical_fault_given_big_cd_bit=(logical_given_bit, 0.0),
            phase_backaction_scale=(0.01, 0.01),
            small_cd_bit_backaction_scale=(0.02, 0.02),
            misclassification_rotation_max_rad=0.6,
            parameter_provenance="T2.2.2 seeded validation assumptions",
        ),
    )
    sbs_trajectory = sbs.simulate(("K_gg",) * samples, seed=normalized_seed)
    big_bit = np.fromiter(
        (
            any(
                event.constituent == "X"
                and event.fault_type == "bit_flip"
                and event.stage == "big_cd"
                for event in step.fault_truth.events
            )
            for step in sbs_trajectory.steps
        ),
        dtype=bool,
        count=samples,
    )
    logical = np.fromiter(
        (step.fault_truth.logical_backaction_by_constituent[0] for step in sbs_trajectory.steps),
        dtype=bool,
        count=samples,
    )
    phase_toggle = np.fromiter(
        (
            any(
                event.fault_type == "phase_flip"
                and event.toggles_z_basis_outcome
                for event in step.fault_truth.events
            )
            for step in sbs_trajectory.steps
        ),
        dtype=bool,
        count=samples,
    )

    phase_probability = 0.04
    readout_error = 0.015
    bit_probability = 0.02
    sharp = SharpenTrimAncillaModel(
        SharpenTrimAncillaConfig(
            bit_flip_probabilities=(bit_probability,) * 4,
            phase_flip_probabilities=(phase_probability,) * 4,
            leakage_injection_probabilities=(0.0,) * 4,
            readout_confusion=np.asarray(
                [
                    [1.0 - readout_error, readout_error],
                    [readout_error, 1.0 - readout_error],
                    [0.5, 0.5],
                ]
            ),
            correct_reset_success=(1.0, 1.0),
            wrong_sign_reset_success=1.0,
            leakage_reset_success=0.0,
            peak_feedback_fraction=0.08,
            peak_feedback_asymmetry_fraction=0.0,
            trim_feedback_fraction=0.5,
            lattice=LATTICE_CONST,
            counter_max=2**31 - 1,
            parameter_provenance="T2.2.2 seeded validation assumptions",
            readout_provenance="T2.2.2 symmetric binary confusion assumption",
            reset_provenance="T2.2.2 perfect reset isolation assumption",
        )
    )
    sharp_trajectory = sharp.simulate(("+y",) * samples, seed=normalized_seed + 1)
    mismatch = np.fromiter(
        (item.feedback_direction_wrong for item in sharp_trajectory.truth_rounds),
        dtype=bool,
        count=samples,
    )
    bit_middle = np.fromiter(
        (
            item.bit_flip
            and item.bit_flip_interaction_fraction is not None
            and 0.25 <= item.bit_flip_interaction_fraction <= 0.75
            for item in sharp_trajectory.truth_rounds
        ),
        dtype=bool,
        count=samples,
    )
    bit_logical = np.fromiter(
        (item.logical_backaction != "I" for item in sharp_trajectory.truth_rounds),
        dtype=bool,
        count=samples,
    )
    # bit 与 phase 都翻转 Y outcome；两者同时发生会抵消。随后再经过 symmetric
    # binary readout confusion。
    toggle_probability = (
        bit_probability * (1.0 - phase_probability)
        + (1.0 - bit_probability) * phase_probability
    )
    expected_mismatch = (
        toggle_probability * (1.0 - readout_error)
        + (1.0 - toggle_probability) * readout_error
    )
    sbs_bit_observed = float(np.mean(big_bit))
    sbs_logical_observed = float(np.mean(logical))
    sharp_mismatch_observed = float(np.mean(mismatch))
    standard_error_bit = np.sqrt(bit_rate * (1.0 - bit_rate) / samples)
    standard_error_logical = np.sqrt(
        bit_rate * logical_given_bit * (1.0 - bit_rate * logical_given_bit) / samples
    )
    standard_error_mismatch = np.sqrt(
        expected_mismatch * (1.0 - expected_mismatch) / samples
    )
    checks = MappingProxyType(
        {
            "sbs_big_cd_bit_rate_within_5se": abs(sbs_bit_observed - bit_rate)
            <= 5.0 * standard_error_bit,
            "sbs_logical_backaction_within_5se": abs(
                sbs_logical_observed - bit_rate * logical_given_bit
            )
            <= 5.0 * standard_error_logical,
            "sbs_phase_does_not_toggle_z_basis": not bool(np.any(phase_toggle)),
            "sharpen_feedback_mismatch_within_5se": abs(
                sharp_mismatch_observed - expected_mismatch
            )
            <= 5.0 * standard_error_mismatch,
            "sharpen_bit_middle_window_is_half": abs(
                float(np.mean(bit_middle)) - 0.5 * bit_probability
            )
            <= 5.0 * standard_error_logical,
            "secondary_protocols_remain_non_executable": all(
                not spec.executable for spec in secondary_protocol_noise_specs()
            ),
        }
    )
    return ProtocolAncillaValidationResult(
        samples=samples,
        seed=normalized_seed,
        sbs_expected_big_cd_bit_rate=bit_rate,
        sbs_observed_big_cd_bit_rate=sbs_bit_observed,
        sbs_expected_logical_backaction_rate=bit_rate * logical_given_bit,
        sbs_observed_logical_backaction_rate=sbs_logical_observed,
        sbs_phase_outcome_toggle_rate=float(np.mean(phase_toggle)),
        sharpen_expected_feedback_mismatch_rate=expected_mismatch,
        sharpen_observed_feedback_mismatch_rate=sharp_mismatch_observed,
        sharpen_bit_middle_window_rate=float(np.mean(bit_middle)),
        sharpen_bit_logical_backaction_rate=float(np.mean(bit_logical)),
        checks=checks,
    )


def write_protocol_ancilla_validation(
    result: ProtocolAncillaValidationResult,
    output_path: str | Path,
) -> Path:
    if not isinstance(result, ProtocolAncillaValidationResult):
        raise TypeError("result must be a ProtocolAncillaValidationResult")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path

