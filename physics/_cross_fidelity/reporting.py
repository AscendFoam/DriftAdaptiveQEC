"""Private reporting and CLI for cross-fidelity validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cross_fidelity_validation import (
        CrossFidelityPoint,
        CrossFidelityValidationResult,
        ErrorAttribution,
        _ComparisonGaps,
        _DirectionalTrends,
    )


def _error_attributions(
    comparison: _ComparisonGaps,
    trends: _DirectionalTrends,
    canonical_qp_gap: float,
    legacy_p_audit_gap: float,
) -> tuple[ErrorAttribution, ...]:
    from ..cross_fidelity_validation import ErrorAttribution

    return (
        ErrorAttribution(
            attribution_id="XA-LOW-CLIPPING",
            region="3 dB",
            observation=(
                f"noise-transfer versus direct-syndrome q-LER gap is {comparison.low_noise_syndrome:.6f}; "
                f"clipping ratio is {comparison.low.noise_transfer.minimum_clipping_ratio:.6f}"
            ),
            primary_cause="localized Gaussian peak assumption loses state/envelope structure under clipping",
            reporting_consequence="3/5 dB remain falsification cases, never calibration points for the surrogate",
        ),
        ErrorAttribution(
            attribution_id="XA-HIGH-CUTOFF",
            region="12 dB",
            observation=(
                f"cutoff-24 to cutoff-48 q-LER changes from {trends.cutoff_ler[0]:.6g} "
                f"to {trends.cutoff_ler[-1]:.6g} while minimum capture rises from "
                f"{trends.cutoff_capture[0]:.6f} to {trends.cutoff_capture[-1]:.6f}"
            ),
            primary_cause="narrow high-squeezing peaks require larger photon-number cutoff for tail-sensitive alias rates",
            reporting_consequence="12 dB Fock agreement uses an absolute gate and retains the full cutoff sweep",
        ),
        ErrorAttribution(
            attribution_id="XA-P-COORDINATE",
            region="all, strongest at 10/12 dB",
            observation=(
                f"maximum high-squeezing canonical |p-q| LER gap is {canonical_qp_gap:.6g}; "
                f"the frozen legacy ambiguous audit still has minimum p-q gap "
                f"{legacy_p_audit_gap:.6f}"
            ),
            primary_cause=(
                "the former audit identified decoder logical-cell spacing sqrt(2*pi) with a "
                "canonical Fock domain and omitted the width/envelope/Jacobian dilation"
            ),
            reporting_consequence=(
                "axis-resolved canonical q/p responses are allowed; the independent-axis Pauli "
                "projection is not promoted to a coherent joint-axis correlation claim"
            ),
        ),
        ErrorAttribution(
            attribution_id="XA-OCCUPANCY-SEMANTICS",
            region="all",
            observation="Fock code survival, Gaussian central-domain mass and Pauli correct-coset occupancy have different denominators",
            primary_cause="protocol leakage, alias-domain localization and logical-coset correctness are distinct events",
            reporting_consequence="only directional native-occupancy trends are compared; absolute occupancy values are never ranked",
        ),
    )


def _validation_checks(
    points: list[CrossFidelityPoint],
    comparison: _ComparisonGaps,
    trends: _DirectionalTrends,
    canonical_qp_gap: float,
    legacy_p_audit_gap: float,
    attributions: tuple[ErrorAttribution, ...],
) -> dict[str, bool]:
    from ..cross_fidelity_validation import _strictly_decreasing, _strictly_increasing

    return {
        "all_common_lanes_have_consistent_LER_occupancy_Favg_directions": trends.common_lanes_consistent,
        "fock_protocol_survival_improves_with_squeezing": _strictly_increasing(
            trends.protocol_survival
        ),
        "fock_protocol_code_weighted_fidelity_improves_with_squeezing": _strictly_increasing(
            trends.protocol_weighted
        ),
        "effective_central_domain_occupancy_improves": _strictly_increasing(
            trends.effective_occupancy
        ),
        "noise_transfer_central_domain_occupancy_improves": _strictly_increasing(
            trends.noise_occupancy
        ),
        "high_squeezing_fock_q_matches_direct_syndrome_absolutely": comparison.fock_syndrome < 5.0e-4,
        "high_squeezing_noise_transfer_matches_direct_syndrome": comparison.noise_syndrome < 1.0e-4,
        "high_squeezing_effective_MC_matches_noise_transfer": comparison.effective_noise_z < 5.0,
        "low_squeezing_clipping_mismatch_is_exposed": comparison.low_noise_syndrome > 1.0e-2
        and comparison.low.noise_transfer.validity == "clipping_dominated",
        "high_squeezing_noise_transfer_is_localized": all(
            item.noise_transfer.validity == "localized" for item in comparison.high
        ),
        "twelve_db_fock_q_alias_converges_with_cutoff": _strictly_decreasing(
            trends.cutoff_ler
        )
        and _strictly_increasing(trends.cutoff_capture),
        "fock_reconstructed_probability_mass_is_complete": max(
            item.fock.maximum_reconstructed_mass_error for item in points
        )
        < 1.0e-10,
        "canonical_fock_qp_fourier_alignment_is_restored": canonical_qp_gap
        < 1.0e-5,
        "legacy_ambiguous_fourier_mismatch_is_retained_as_negative_provenance": legacy_p_audit_gap
        > 0.1,
        "error_attribution_table_is_complete": len(attributions) >= 4,
    }


def _write_cross_fidelity_validation(
    result: CrossFidelityValidationResult,
    output: str | Path,
) -> Path:
    from ..cross_fidelity_validation import CrossFidelityValidationResult

    if not isinstance(result, CrossFidelityValidationResult):
        raise TypeError("result must be a CrossFidelityValidationResult")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["passed"] = result.passed
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def main() -> int:
    from .. import cross_fidelity_validation as validation

    parser = argparse.ArgumentParser(description=validation.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2026071433)
    arguments = parser.parse_args()
    result = validation.run_cross_fidelity_validation(
        validation.CrossFidelityConfig(
            effective_samples=arguments.effective_samples,
            seed=arguments.seed,
        )
    )
    validation.write_cross_fidelity_validation(result, arguments.output)
    print(json.dumps({"passed": result.passed, "checks": result.checks}, sort_keys=True))
    return 0 if result.passed else 1
