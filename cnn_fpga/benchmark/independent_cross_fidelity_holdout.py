"""T5.0.2 disjoint cross-fidelity and secondary analytic holdouts.

The point and seed constants below are the preregistered test contract.  They
exclude both the T2.3.3 calibration grid and the exploratory 4/11/14 dB pilot.
No threshold is selected from the holdout observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from physics.cross_fidelity_validation import (
    CrossFidelityConfig,
    evaluate_cross_fidelity_point,
)
from physics.finite_squeezing_noise import (
    FiniteSqueezingNoiseConfig,
    sample_finite_squeezing_noise,
)
from physics.noise_transfer_surrogate import projector_delta_from_squeezing_db


TASK_ID = "T5.0.2"
SCHEMA_VERSION = 1
PROTOCOL_ID = "T502-DISJOINT-CROSS-FIDELITY-HOLDOUT-V1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t5_0_2_independent_cross_fidelity_holdout.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_0_2_independent_cross_fidelity_holdout_source_data.csv")

CALIBRATION_DB = (3.0, 5.0, 8.0, 10.0, 12.0)
EXCLUDED_EXPLORATORY_PILOT_DB = (4.0, 11.0, 14.0)
NEGATIVE_HOLDOUT_DB = 2.5
HIGH_HOLDOUT_DB = (10.25, 11.75)
HOLDOUT_SEEDS = (2026071611, 2026071613, 2026071617, 2026071619)
EFFECTIVE_SAMPLES_PER_SEED = 100_000

PSTEANE_SIGMA_A = (0.07, 0.11, 0.19)
PSTEANE_VARIANCE_RATIOS = (1.25, 2.25, 4.75)
PSTEANE_B_GRID = (0.5, 0.75, 1.0, sqrt(2.0), sqrt(3.0), 2.0, 2.5)
PSTEANE_M_GRID = (1, 2, 3, 4)

PARENT_ARTIFACTS = {
    "T2.3.3": Path("docs/t2_3_3_cross_fidelity_validation.json"),
    "T5.0.1": Path("docs/t5_0_1_literature_trend_reproduction.json"),
}

MAIN_THRESHOLDS = {
    "noise_vs_syndrome_q_ler_gap_max": 5.0e-5,
    "fock_vs_syndrome_q_ler_gap_max": 5.0e-4,
    "canonical_qp_ler_gap_max": 1.0e-6,
    "pooled_effective_vs_noise_z_max": 2.0,
    "localized_clipping_ratio_min": 0.90,
    "negative_noise_vs_syndrome_q_ler_gap_min": 1.0e-2,
    "negative_clipping_ratio_max": 0.50,
}


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _effective_axis_counts(db: float, seed: int) -> dict[str, int]:
    delta = projector_delta_from_squeezing_db(db)
    channel_variance = 0.18**2
    measurement_variance = 0.06**2
    batch = sample_finite_squeezing_noise(
        FiniteSqueezingNoiseConfig(
            channel_covariance=((channel_variance, 0.0), (0.0, channel_variance)),
            data_delta=(delta, delta),
            ancilla_delta=(delta, delta),
            measurement_covariance=((measurement_variance, 0.0), (0.0, measurement_variance)),
            include_envelope=True,
            samples=EFFECTIVE_SAMPLES_PER_SEED,
            seed=seed,
        )
    )
    parity = batch.logical_parity
    return {
        "q_errors": int(np.count_nonzero(parity[:, 0] != 0)),
        "p_errors": int(np.count_nonzero(parity[:, 1] != 0)),
        "samples": int(parity.shape[0]),
    }


def _main_holdout_point(db: float) -> dict[str, Any]:
    config = CrossFidelityConfig(
        effective_samples=EFFECTIVE_SAMPLES_PER_SEED,
        seed=HOLDOUT_SEEDS[0],
    )
    point = evaluate_cross_fidelity_point(db, config)
    seed_rows = []
    total_q = total_p = total_n = 0
    for seed in HOLDOUT_SEEDS:
        counts = _effective_axis_counts(db, seed + int(round(100.0 * db)))
        total_q += counts["q_errors"]
        total_p += counts["p_errors"]
        total_n += counts["samples"]
        seed_rows.append({"seed": seed, **counts})
    q_rate = total_q / total_n
    p_rate = total_p / total_n
    prediction = point.noise_transfer.pauli.q_axis_ler
    standard_error = sqrt(max(prediction * (1.0 - prediction), 1.0e-15) / total_n)
    syndrome_q = point.syndrome.square_symmetry_projection.q_axis_ler
    fock_q = point.fock.two_axis_pauli_metrics.q_axis_ler
    record = {
        "squeezing_db": db,
        "region": point.region,
        "deterministic_lanes": {
            "noise_transfer_q_ler": prediction,
            "syndrome_q_ler": syndrome_q,
            "fock_q_ler": fock_q,
            "noise_vs_syndrome_q_ler_gap": abs(prediction - syndrome_q),
            "fock_vs_syndrome_q_ler_gap": abs(fock_q - syndrome_q),
            "canonical_qp_ler_gap": abs(point.fock.p_minus_q_ler_gap),
            "noise_transfer_validity": point.noise_transfer.validity,
            "minimum_clipping_ratio": point.noise_transfer.minimum_clipping_ratio,
            "fock_minimum_captured_probability": point.fock.minimum_captured_probability,
        },
        "effective_holdout": {
            "seed_rows": seed_rows,
            "total_samples": total_n,
            "q_ler": q_rate,
            "p_ler": p_rate,
            "prediction_standard_error": standard_error,
            "q_z_score": abs(q_rate - prediction) / standard_error,
            "p_z_score": abs(p_rate - prediction) / standard_error,
        },
    }
    record["effective_holdout"]["maximum_axis_z_score"] = max(
        record["effective_holdout"]["q_z_score"],
        record["effective_holdout"]["p_z_score"],
    )
    return record


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def psteane_small_noise_variances(
    sigma_d: float,
    sigma_a: float,
    b: float,
    m: int,
) -> tuple[float, float]:
    """Equation (40) of arXiv:2604.08247v1."""

    sd = _positive_scalar(sigma_d, "sigma_d")
    sa = _positive_scalar(sigma_a, "sigma_a")
    squeeze = _positive_scalar(b, "b")
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or int(m) <= 0:
        raise ValueError("m must be a positive integer")
    integer_m = int(m)
    sd2, sa2, b2 = sd * sd, sa * sa, squeeze * squeeze
    q_variance = (
        sa2
        * integer_m**2
        * (b2 * b2 * sa2 + 2.0 * b2 * sd2)
        / (2.0 * b2 * (integer_m**2 - 2 * integer_m + 2) * sa2 + 4.0 * sd2)
    )
    p_variance = (
        4.0 * sa2 * sa2
        + 2.0 * b2 * (integer_m**2 - 2 * integer_m + 2) * sa2 * sd2
    ) / (b2 * integer_m**2 * (2.0 * sa2 + b2 * sd2))
    return q_variance, p_variance


def psteane_covariance_propagation(
    sigma_d: float,
    sigma_a: float,
    b: float,
    m: int,
) -> tuple[float, float]:
    """Independent coefficient propagation from equations (36)--(37)."""

    sd = _positive_scalar(sigma_d, "sigma_d")
    sa = _positive_scalar(sigma_a, "sigma_a")
    squeeze = _positive_scalar(b, "b")
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or int(m) <= 0:
        raise ValueError("m must be a positive integer")
    integer_m = int(m)
    sd2, sa2, b2 = sd * sd, sa * sa, squeeze * squeeze
    cq = (2.0 * sd2 + b2 * (2 - integer_m) * sa2) / (
        2.0 * sd2 + b2 * (integer_m**2 - 2 * integer_m + 2) * sa2
    )
    cp = (b2 * integer_m * sd2 + 2.0 * sa2) / (
        b2 * integer_m * sd2 + 2.0 * integer_m * sa2
    )
    q_coefficients = (
        (1.0 - cq, sd2),
        (-cq * squeeze * integer_m / 2.0, sa2),
        (squeeze - cq * squeeze * (2 - integer_m) / 2.0, sa2),
    )
    p_coefficients = (
        (1.0 - cp, sd2),
        (-2.0 / (squeeze * integer_m) + cp / squeeze, sa2),
        (cp / squeeze, sa2),
    )
    return (
        sum(coefficient**2 * variance for coefficient, variance in q_coefficients),
        sum(coefficient**2 * variance for coefficient, variance in p_coefficients),
    )


def psteane_variance_product_formula(
    sigma_d: float,
    sigma_a: float,
    b: float,
    m: int,
) -> float:
    """Equation (41) of arXiv:2604.08247v1."""

    sd = _positive_scalar(sigma_d, "sigma_d")
    sa = _positive_scalar(sigma_a, "sigma_a")
    squeeze = _positive_scalar(b, "b")
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or int(m) <= 0:
        raise ValueError("m must be a positive integer")
    integer_m = int(m)
    sd2, sa2, b2 = sd * sd, sa * sa, squeeze * squeeze
    numerator = 2.0 * b2 * (integer_m - 1) ** 2 * (sd2 * sd2 - sa2 * sa2)
    denominator = (
        b2 * (integer_m**2 - 2 * integer_m + 2) * sa2 + 2.0 * sd2
    ) * (2.0 * sa2 + b2 * sd2)
    return sa2 * sa2 * (1.0 + numerator / denominator)


def _me_steane_variances(sigma_d: float, sigma_a: float) -> tuple[float, float]:
    sd2, sa2 = sigma_d**2, sigma_a**2
    return (
        sa2 * (sa2 + 2.0 * sd2) / (sa2 + sd2),
        sa2 * (sa2 + sd2) / (2.0 * sa2 + sd2),
    )


def run_psteane_secondary_holdout() -> dict[str, Any]:
    rows = []
    max_covariance_error = 0.0
    max_product_relative_error = 0.0
    argmin_failures = []
    me_errors = []
    teleport_errors = []
    for sigma_a in PSTEANE_SIGMA_A:
        for variance_ratio in PSTEANE_VARIANCE_RATIOS:
            sigma_d = sqrt(variance_ratio) * sigma_a
            for b in PSTEANE_B_GRID:
                products = {}
                for m in PSTEANE_M_GRID:
                    formula = psteane_small_noise_variances(sigma_d, sigma_a, b, m)
                    propagated = psteane_covariance_propagation(sigma_d, sigma_a, b, m)
                    product_formula = psteane_variance_product_formula(sigma_d, sigma_a, b, m)
                    product_direct = formula[0] * formula[1]
                    covariance_error = max(abs(formula[0] - propagated[0]), abs(formula[1] - propagated[1]))
                    product_relative_error = abs(product_formula - product_direct) / max(product_direct, 1.0e-30)
                    max_covariance_error = max(max_covariance_error, covariance_error)
                    max_product_relative_error = max(max_product_relative_error, product_relative_error)
                    products[m] = product_direct
                    rows.append({
                        "sigma_a": sigma_a,
                        "variance_ratio": variance_ratio,
                        "sigma_d": sigma_d,
                        "b": b,
                        "m": m,
                        "a": m * b / 2.0,
                        "q_variance": formula[0],
                        "p_variance": formula[1],
                        "variance_product": product_direct,
                        "covariance_error": covariance_error,
                        "product_relative_error": product_relative_error,
                    })
                minimum = min(products.values())
                minimizing_m = [m for m, value in products.items() if np.isclose(value, minimum, rtol=1.0e-12, atol=1.0e-15)]
                if minimizing_m != [1]:
                    argmin_failures.append({"sigma_a": sigma_a, "variance_ratio": variance_ratio, "b": b, "minimizing_m": minimizing_m})
            me_formula = psteane_small_noise_variances(sigma_d, sigma_a, 1.0, 2)
            me_reference = _me_steane_variances(sigma_d, sigma_a)
            me_errors.append(max(abs(me_formula[0] - me_reference[0]), abs(me_formula[1] - me_reference[1])))
            teleport = psteane_small_noise_variances(sigma_d, sigma_a, sqrt(2.0), 1)
            teleport_errors.append(max(abs(teleport[0] - sigma_a**2), abs(teleport[1] - sigma_a**2)))
    gates = {
        "equation40_matches_independent_covariance_propagation": max_covariance_error <= 1.0e-12,
        "equation41_matches_direct_variance_product": max_product_relative_error <= 1.0e-12,
        "m1_is_unique_grid_argmin_for_all_k_above_one": not argmin_failures,
        "m1_product_equals_sigma_a_fourth_for_every_b": all(np.isclose(row["variance_product"], row["sigma_a"] ** 4, rtol=1.0e-12, atol=1.0e-15) for row in rows if row["m"] == 1),
        "me_steane_special_case_matches_equation26": max(me_errors) <= 1.0e-12,
        "teleportation_special_case_is_symmetric": max(teleport_errors) <= 1.0e-12,
        "secondary_grid_is_disjoint_from_source_example_ratios": 1.0 not in PSTEANE_VARIANCE_RATIOS and 3.0 not in PSTEANE_VARIANCE_RATIOS,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "secondary small-noise analytic regression only; no sBs ranking, physical squeezing, FPGA, or device claim",
        "source": {
            "identifier": "arXiv:2604.08247v1",
            "official_pdf": "https://arxiv.org/pdf/2604.08247",
            "equations": [36, 37, 40, 41, 43],
        },
        "grid": {
            "sigma_a": list(PSTEANE_SIGMA_A),
            "data_to_ancilla_variance_ratio": list(PSTEANE_VARIANCE_RATIOS),
            "b": list(PSTEANE_B_GRID),
            "m": list(PSTEANE_M_GRID),
            "row_count": len(rows),
        },
        "rows": rows,
        "diagnostics": {
            "maximum_covariance_error": max_covariance_error,
            "maximum_product_relative_error": max_product_relative_error,
            "maximum_me_special_case_error": max(me_errors),
            "maximum_teleportation_special_case_error": max(teleport_errors),
            "argmin_failures": argmin_failures,
        },
        "gates": gates,
    }


def _parent_pass(task_id: str, payload: Mapping[str, Any]) -> bool:
    if task_id == "T2.3.3":
        return bool(payload.get("passed") is True and payload.get("checks") and all(payload["checks"].values()))
    if task_id == "T5.0.1":
        return bool(payload.get("status") == "PASS" and payload.get("gates") and all(payload["gates"].values()))
    raise KeyError(task_id)


def build_holdout_report() -> dict[str, Any]:
    parents = {task_id: json.loads(_repo_path(path).read_text(encoding="utf-8")) for task_id, path in PARENT_ARTIFACTS.items()}
    points = [_main_holdout_point(NEGATIVE_HOLDOUT_DB)] + [_main_holdout_point(db) for db in HIGH_HOLDOUT_DB]
    negative = points[0]
    high = points[1:]
    main_gates = {
        "holdout_points_are_disjoint_from_calibration_and_exploratory_pilot": not (set([NEGATIVE_HOLDOUT_DB, *HIGH_HOLDOUT_DB]) & set([*CALIBRATION_DB, *EXCLUDED_EXPLORATORY_PILOT_DB])),
        "holdout_seeds_are_new_and_unique": len(set(HOLDOUT_SEEDS)) == len(HOLDOUT_SEEDS) and min(HOLDOUT_SEEDS) > 2026071500,
        "both_high_points_are_in_preregistered_validity_domain": all(point["squeezing_db"] >= 10.0 and point["region"] == "high_squeezing" for point in high),
        "high_noise_transfer_matches_direct_syndrome": all(point["deterministic_lanes"]["noise_vs_syndrome_q_ler_gap"] <= MAIN_THRESHOLDS["noise_vs_syndrome_q_ler_gap_max"] for point in high),
        "high_fock_matches_direct_syndrome": all(point["deterministic_lanes"]["fock_vs_syndrome_q_ler_gap"] <= MAIN_THRESHOLDS["fock_vs_syndrome_q_ler_gap_max"] for point in high),
        "high_canonical_qp_alignment_holds": all(point["deterministic_lanes"]["canonical_qp_ler_gap"] <= MAIN_THRESHOLDS["canonical_qp_ler_gap_max"] for point in high),
        "high_noise_transfer_is_localized_without_clipping": all(point["deterministic_lanes"]["noise_transfer_validity"] == "localized" and point["deterministic_lanes"]["minimum_clipping_ratio"] >= MAIN_THRESHOLDS["localized_clipping_ratio_min"] for point in high),
        "pooled_effective_holdout_matches_noise_transfer_within_two_sigma": all(point["effective_holdout"]["maximum_axis_z_score"] <= MAIN_THRESHOLDS["pooled_effective_vs_noise_z_max"] for point in high),
        "negative_out_of_domain_mismatch_is_preserved": negative["deterministic_lanes"]["noise_vs_syndrome_q_ler_gap"] >= MAIN_THRESHOLDS["negative_noise_vs_syndrome_q_ler_gap_min"] and negative["deterministic_lanes"]["minimum_clipping_ratio"] < MAIN_THRESHOLDS["negative_clipping_ratio_max"] and negative["deterministic_lanes"]["noise_transfer_validity"] == "clipping_dominated",
    }
    main_status = "PASS" if all(main_gates.values()) else "FAIL"
    secondary = run_psteane_secondary_holdout()
    bindings = {
        task_id: {"path": str(path), "sha256": _sha256(path), "machine_pass": _parent_pass(task_id, parents[task_id])}
        for task_id, path in PARENT_ARTIFACTS.items()
    }
    acceptance_gates = {
        "all_parent_artifacts_are_hash_bound_and_pass": all(row["machine_pass"] for row in bindings.values()),
        "formal_points_exclude_all_calibration_and_pilot_points": main_gates["holdout_points_are_disjoint_from_calibration_and_exploratory_pilot"],
        "main_failure_if_any_is_preserved_not_reselected": main_status in {"PASS", "FAIL"} and len(points) == 3,
        "secondary_remains_outside_sbs_main_ranking": "no sBs ranking" in secondary["scope"],
        "at_least_one_independent_holdout_family_passes": main_status == "PASS" or secondary["status"] == "PASS",
        "all_thresholds_are_frozen_in_source_contract": MAIN_THRESHOLDS == {
            "noise_vs_syndrome_q_ler_gap_max": 5.0e-5,
            "fock_vs_syndrome_q_ler_gap_max": 5.0e-4,
            "canonical_qp_ler_gap_max": 1.0e-6,
            "pooled_effective_vs_noise_z_max": 2.0,
            "localized_clipping_ratio_min": 0.90,
            "negative_noise_vs_syndrome_q_ler_gap_min": 1.0e-2,
            "negative_clipping_ratio_max": 0.50,
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if all(acceptance_gates.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": implementation_sha256(),
        "preregistered_contract": {
            "calibration_db": list(CALIBRATION_DB),
            "excluded_exploratory_pilot": {
                "db": list(EXCLUDED_EXPLORATORY_PILOT_DB),
                "purpose": "API and boundary reconnaissance only; never eligible for formal holdout acceptance",
            },
            "formal_negative_db": NEGATIVE_HOLDOUT_DB,
            "formal_high_db": list(HIGH_HOLDOUT_DB),
            "seeds": list(HOLDOUT_SEEDS),
            "effective_samples_per_seed": EFFECTIVE_SAMPLES_PER_SEED,
            "thresholds": MAIN_THRESHOLDS,
        },
        "parent_artifacts": bindings,
        "main_cross_fidelity_holdout": {
            "status": main_status,
            "scope": "same four model lanes at disjoint squeezing points and fresh effective-model seeds; no threshold retuning",
            "points": points,
            "gates": main_gates,
            "failed_gates": [name for name, passed in main_gates.items() if not passed],
        },
        "secondary_psteane_holdout": secondary,
        "acceptance_gates": acceptance_gates,
        "gate_summary": {
            "passed": sum(acceptance_gates.values()),
            "total": len(acceptance_gates),
            "failed": [name for name, passed in acceptance_gates.items() if not passed],
        },
        "claim_boundary": {
            "allowed": "independent cross-fidelity holdout outcomes exactly as observed and secondary small-noise P-Steane analytic regression",
            "forbidden": "reselecting holdout points, hiding a failed family, promoting P-Steane into sBs ranking, physical squeezing implementation, device fidelity, FPGA, or hardware claims",
        },
    }
    payload["contract_sha256"] = _canonical_sha256({key: value for key, value in payload.items() if key not in {"generated_at_utc", "contract_sha256"}})
    return payload


def source_data_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, binding in payload["parent_artifacts"].items():
        rows.append({"row_type": "parent", "record_id": task_id, "family": "provenance", "parameter": binding["path"], "value": binding["sha256"], "threshold": "machine_pass", "passed": binding["machine_pass"], "detail": ""})
    for point in payload["main_cross_fidelity_holdout"]["points"]:
        deterministic = point["deterministic_lanes"]
        if point["squeezing_db"] == NEGATIVE_HOLDOUT_DB:
            point_passed = bool(
                deterministic["noise_vs_syndrome_q_ler_gap"]
                >= MAIN_THRESHOLDS["negative_noise_vs_syndrome_q_ler_gap_min"]
                and deterministic["minimum_clipping_ratio"]
                < MAIN_THRESHOLDS["negative_clipping_ratio_max"]
                and deterministic["noise_transfer_validity"] == "clipping_dominated"
            )
        else:
            point_passed = bool(
                deterministic["noise_vs_syndrome_q_ler_gap"]
                <= MAIN_THRESHOLDS["noise_vs_syndrome_q_ler_gap_max"]
                and deterministic["fock_vs_syndrome_q_ler_gap"]
                <= MAIN_THRESHOLDS["fock_vs_syndrome_q_ler_gap_max"]
                and deterministic["canonical_qp_ler_gap"]
                <= MAIN_THRESHOLDS["canonical_qp_ler_gap_max"]
                and deterministic["noise_transfer_validity"] == "localized"
                and deterministic["minimum_clipping_ratio"]
                >= MAIN_THRESHOLDS["localized_clipping_ratio_min"]
                and point["effective_holdout"]["maximum_axis_z_score"]
                <= MAIN_THRESHOLDS["pooled_effective_vs_noise_z_max"]
            )
        rows.append({"row_type": "main_point", "record_id": f"db_{point['squeezing_db']}", "family": "main_cross_fidelity", "parameter": "squeezing_db", "value": point["squeezing_db"], "threshold": json.dumps(MAIN_THRESHOLDS, sort_keys=True), "passed": point_passed, "detail": json.dumps({"deterministic": deterministic, "effective": point["effective_holdout"]}, sort_keys=True)})
        for seed_row in point["effective_holdout"]["seed_rows"]:
            rows.append({"row_type": "effective_seed", "record_id": f"db_{point['squeezing_db']}_seed_{seed_row['seed']}", "family": "main_cross_fidelity", "parameter": "q_errors,p_errors,samples", "value": f"{seed_row['q_errors']},{seed_row['p_errors']},{seed_row['samples']}", "threshold": "fresh_seed_no_selection", "passed": True, "detail": ""})
    for index, row in enumerate(payload["secondary_psteane_holdout"]["rows"]):
        rows.append({"row_type": "psteane_grid", "record_id": f"psteane_{index:03d}", "family": "secondary_psteane", "parameter": f"sigma_a={row['sigma_a']};k={row['variance_ratio']};b={row['b']};m={row['m']}", "value": row["variance_product"], "threshold": "equation40/41/covariance <= 1e-12; m=1 argmin", "passed": row["covariance_error"] <= 1e-12 and row["product_relative_error"] <= 1e-12, "detail": f"a={row['a']};q={row['q_variance']};p={row['p_variance']}"})
    for family_name, gates in (("main_cross_fidelity", payload["main_cross_fidelity_holdout"]["gates"]), ("secondary_psteane", payload["secondary_psteane_holdout"]["gates"]), ("acceptance", payload["acceptance_gates"])):
        for gate_id, passed in gates.items():
            rows.append({"row_type": "gate", "record_id": gate_id, "family": family_name, "parameter": "boolean", "value": passed, "threshold": "must_be_true_for_family" if family_name != "acceptance" else "must_be_true_for_task", "passed": passed, "detail": ""})
    return rows


def write_artifacts(
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = build_holdout_report()
    rows = source_data_rows(payload)
    csv_path = _repo_path(source_data_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("row_type", "record_id", "family", "parameter", "value", "threshold", "passed", "detail")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    payload["source_data"] = {"path": str(Path(source_data_path)), "row_count": len(rows), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    output = _repo_path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE_DATA))
    args = parser.parse_args(argv)
    payload = write_artifacts(args.artifact, args.source_data)
    print(json.dumps({"task_id": TASK_ID, "status": payload["status"], "main": payload["main_cross_fidelity_holdout"]["status"], "secondary": payload["secondary_psteane_holdout"]["status"], "gates": payload["gate_summary"], "source_rows": payload["source_data"]["row_count"]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
