"""T6.8.1 same-model static GKP decoder comparison lane.

The lane consumes the frozen T6.7.1 smooth formal counts and adds a top-K
implementation result without rerunning or sampling the formal data.  This is
possible because every deployable input is one of 1024x1024 quantized syndrome
bin-centre pairs: exhaustive action equivalence on that entire domain lets the
K=4 row inherit the full-static Pauli/error trace exactly.  Soft-output error is
reported separately and is never hidden by hard-decision equivalence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from cnn_fpga.benchmark import route_a_smooth_formal as smooth
from cnn_fpga.benchmark.route_a_posterior_calibration import _load_static_and_hyperparameters
from cnn_fpga.benchmark.topk_lattice_coset_map import (
    _full_axis_llrs,
    topk_cost_profile,
    topk_map_decode_2d,
)
from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import map_decode_2d


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.8.1"
SCHEMA_VERSION = "t6.8.1-static-gkp-same-model-lane-v1"
VERDICT = "PASS_STATIC_GKP_SAME_MODEL_LANE_ROUTE_A_SUPERIORITY_FALSIFIED"
PARENT = ROOT / "docs" / "t6_7_1_smooth_formal_matrix.json"
TOPK_PARENT = ROOT / "docs" / "t3_1_5_topk_map_validation.json"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_8_1_static_gkp_same_model_lane.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_8_1_static_gkp_same_model_lane_source_data.csv"
ADC_BITS = 10
ADC_LEVELS = 1 << ADC_BITS
TOPK_K = 4
GRID_CHUNK_Q = 32


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) is None:
            return 0
        return sum(1 for _ in reader)


def _paired_static_contrast(parent: Mapping[str, Any]) -> dict[str, Any]:
    rows = parent["trajectory_results"]
    seeds = [int(value) for value in parent["formal_design"]["seeds"]]
    families = [str(value) for value in parent["formal_design"]["families"]]

    def seed_values(method: str) -> np.ndarray:
        output = []
        for seed in seeds:
            family_values = []
            for family in families:
                selected = [
                    row for row in rows
                    if int(row["seed"]) == seed and row["family"] == family
                ]
                if len(selected) != 6:
                    raise ValueError("same-model lane requires six cells per smooth family/seed")
                cell_values = []
                for row in selected:
                    counts = np.asarray(
                        row["method_window_pauli_counts_class_order_I_Z_X_Y"][method],
                        dtype=np.int64,
                    )
                    cell_values.append(float(np.sum(counts[:, 1:]) / np.sum(counts)))
                family_values.append(float(np.mean(cell_values)))
            output.append(float(np.mean(family_values)))
        return np.asarray(output, dtype=np.float64)

    static = seed_values("static_joint_map")
    route = seed_values("proposed_route_a")
    differences = static - route
    indices = np.random.default_rng(smooth.BOOTSTRAP_SEED).integers(
        0, len(seeds), size=(smooth.BOOTSTRAP_REPLICATES, len(seeds))
    )
    samples = np.mean(differences[indices], axis=1)
    return {
        "contrast": "static_full_MAP_LER_minus_Route_A_LER",
        "estimate": float(np.mean(differences)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "one_sided_p_nonpositive": float((1 + np.sum(samples <= 0.0)) / (len(samples) + 1)),
        "route_a_superiority_passes_lcb_gt_zero": bool(np.quantile(samples, 0.025) > 0.0),
        "cluster": "formal seed",
        "clusters": len(seeds),
        "replicates": smooth.BOOTSTRAP_REPLICATES,
        "bootstrap_seed": smooth.BOOTSTRAP_SEED,
    }


def _exhaustive_topk_equivalence() -> dict[str, Any]:
    static, _ = _load_static_and_hyperparameters()
    covariance = static.covariance_array()
    mean = static.mean_array()
    half = 0.5 * LATTICE_CONST
    values = -half + (np.arange(ADC_LEVELS, dtype=np.float64) + 0.5) * LATTICE_CONST / ADC_LEVELS
    topk_hash = hashlib.sha256()
    full_hash = hashlib.sha256()
    q_errors: list[np.ndarray] = []
    p_errors: list[np.ndarray] = []
    disagreement = 0
    chunk_rows: list[dict[str, Any]] = []
    start_time = perf_counter()
    prior = np.full((2, 2), 0.25, dtype=np.float64)
    for q_start in range(0, ADC_LEVELS, GRID_CHUNK_Q):
        q_stop = min(ADC_LEVELS, q_start + GRID_CHUNK_Q)
        syndrome = np.stack(
            (
                np.repeat(values[q_start:q_stop], ADC_LEVELS),
                np.tile(values, q_stop - q_start),
            ),
            axis=1,
        )
        topk = topk_map_decode_2d(
            syndrome,
            covariance,
            mean=mean,
            prior=prior,
            k=TOPK_K,
            tail_sigma=8.0,
        )
        full = map_decode_2d(
            syndrome,
            covariance,
            mean=mean,
            prior=prior,
            tail_sigma=8.0,
        )
        topk_class = np.asarray(topk.logical_class, dtype=np.uint8)
        full_class = np.asarray(full.logical_class, dtype=np.uint8)
        local = int(np.sum(topk_class != full_class))
        disagreement += local
        topk_hash.update(topk_class.tobytes())
        full_hash.update(full_class.tobytes())
        full_q, full_p = _full_axis_llrs(np.asarray(full.log_likelihoods), prior)
        local_q = np.abs(np.asarray(topk.q_llr) - full_q)
        local_p = np.abs(np.asarray(topk.p_llr) - full_p)
        q_errors.append(local_q)
        p_errors.append(local_p)
        chunk_rows.append(
            {
                "q_start": q_start,
                "q_stop": q_stop,
                "points": len(syndrome),
                "hard_disagreements": local,
                "q_llr_max_abs_error": float(np.max(local_q)),
                "p_llr_max_abs_error": float(np.max(local_p)),
            }
        )
    q_error = np.concatenate(q_errors)
    p_error = np.concatenate(p_errors)
    return {
        "adc_bits_per_axis": ADC_BITS,
        "levels_per_axis": ADC_LEVELS,
        "grid_points": ADC_LEVELS * ADC_LEVELS,
        "bin_centres_cover_complete_deployable_domain": True,
        "k": TOPK_K,
        "tail_sigma": 8.0,
        "static_mean": mean.tolist(),
        "static_covariance": covariance.tolist(),
        "hard_disagreements": disagreement,
        "hard_action_equivalent": disagreement == 0,
        "topk_action_sha256": topk_hash.hexdigest(),
        "full_action_sha256": full_hash.hexdigest(),
        "q_llr_p99_abs_error": float(np.quantile(q_error, 0.99)),
        "p_llr_p99_abs_error": float(np.quantile(p_error, 0.99)),
        "q_llr_max_abs_error": float(np.max(q_error)),
        "p_llr_max_abs_error": float(np.max(p_error)),
        "soft_outputs_declared_approximate": True,
        "elapsed_host_seconds_validation_only": perf_counter() - start_time,
        "chunks": chunk_rows,
    }


def _method_table(parent: Mapping[str, Any], equivalence: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = {row["method_id"]: row for row in parent["analysis"]["method_summaries"]}
    costs = parent["analysis"]["cost_ledger"]
    output: list[dict[str, Any]] = []
    for method in ("standard_binning", "static_joint_map", "proposed_route_a", "hidden_state_oracle"):
        source = summaries[method]
        method_cost = deepcopy(costs[method])
        if method == "standard_binning":
            method_cost.update(
                {
                    "decision_operation_proxy": "two centered-cell parity/rounding decisions",
                    "target_measured": False,
                }
            )
        elif method == "static_joint_map":
            method_cost.update(
                {
                    "operation_storage_proxy": vars(topk_cost_profile(
                        np.asarray(equivalence["static_covariance"]),
                        mean=np.asarray(equivalence["static_mean"]),
                        k=128,
                        tail_sigma=8.0,
                    )),
                    "target_measured": False,
                }
            )
        output.append(
            {
                "method_id": method,
                "label": "frozen full periodic Gaussian MAP" if method == "static_joint_map" else method,
                "deployable": bool(source["deployable"]),
                "trace_evidence": "direct T6.7.1 formal counts",
                "decisions": int(source["decisions"]),
                "p_I": float(source["p_I"]),
                "p_X": float(source["p_X"]),
                "p_Y": float(source["p_Y"]),
                "p_Z": float(source["p_Z"]),
                "p_L": float(source["p_L"]),
                "average_ler_equal_family_seed": float(source["average_ler_equal_family_seed"]),
                "p95_window_ler": float(source["p95_window_ler"]),
                "worst_window_ler": float(source["global_worst_window_ler"]),
                "family_ler": source["family_ler_equal_seed_cell"],
                "wallclock_us_per_decision_host_only": float(source["wallclock_us_per_decision"]),
                "cost": method_cost,
            }
        )
    static_row = next(row for row in output if row["method_id"] == "static_joint_map")
    topk = deepcopy(static_row)
    topk.update(
        {
            "method_id": "topk_k4_static_map",
            "label": "K=4 lattice-coset truncated static MAP",
            "trace_evidence": "inherited exactly from full static counts after exhaustive 1024x1024 hard-action equivalence",
            "wallclock_us_per_decision_host_only": None,
            "cost": {
                "update_macs": 0,
                "private_model_state_bytes": 24,
                "transient_workspace_bytes": 256,
                "operation_storage_proxy": vars(topk_cost_profile(
                    np.asarray(equivalence["static_covariance"]),
                    mean=np.asarray(equivalence["static_mean"]),
                    k=TOPK_K,
                    tail_sigma=8.0,
                )),
                "target_measured": False,
            },
        }
    )
    output.insert(2, topk)
    return output


def _literature_registry() -> list[dict[str, Any]]:
    return [
        {
            "role": "standard_binning_origin",
            "title": "Encoding a qubit in an oscillator",
            "year": 2001,
            "doi": "10.1103/PhysRevA.64.012310",
            "primary_url": "https://journals.aps.org/pra/abstract/10.1103/PhysRevA.64.012310",
            "mapping": "nearest-cell GKP correction geometry; this project evaluates the resulting zero-class hard action",
            "numeric_cross_model_comparison_allowed": False,
        },
        {
            "role": "analog_likelihood_precedent",
            "title": "Analog quantum error correction with encoding a qubit into an oscillator",
            "year": 2017,
            "doi": "10.1103/PhysRevLett.119.180507",
            "primary_url": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180507",
            "mapping": "analog residual information informs likelihood-based decoding; outer-code threshold numbers are not imported",
            "numeric_cross_model_comparison_allowed": False,
        },
        {
            "role": "finite_energy_decoding_boundary",
            "title": "Logical channels in approximate Gottesman-Kitaev-Preskill error correction",
            "year": 2025,
            "doi": None,
            "arxiv": "2504.13383",
            "primary_url": "https://arxiv.org/abs/2504.13383",
            "mapping": "supports that finite-energy optimized and standard-binning decoders can differ; its channel numbers are not this syndrome simulator",
            "numeric_cross_model_comparison_allowed": False,
        },
        {
            "role": "prior_calibration_cross_code_boundary",
            "title": "Optimization of Decoder Priors for Accurate Quantum Error Correction",
            "year": 2024,
            "doi": "10.1103/PhysRevLett.133.150603",
            "primary_url": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.150603",
            "mapping": "prior calibration precedent on repetition/surface-code experiments; 16%/3.3% are not GKP benchmarks",
            "numeric_cross_model_comparison_allowed": False,
        },
    ]


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    methods = {row["method_id"]: row for row in report["method_table"]}
    contrast = report["paired_static_contrast"]
    eq = report["topk_full_exhaustive_equivalence"]
    claims = {row["claim_id"]: row["state"] for row in report["claim_registry"]}
    return {
        "G01_parent_formal_report_recomputes": bool(report["parent_audit"]["formal_report_verified"]),
        "G02_exact_same_trace_methods_and_oracle_partition": set(methods) == {"standard_binning", "static_joint_map", "topk_k4_static_map", "proposed_route_a", "hidden_state_oracle"} and not methods["hidden_state_oracle"]["deployable"],
        "G03_all_rows_report_pauli_average_tail_and_family_metrics": all(
            all(key in row for key in ("p_L", "p_X", "p_Y", "p_Z", "average_ler_equal_family_seed", "p95_window_ler", "worst_window_ler", "family_ler", "cost"))
            for row in methods.values()
        ),
        "G04_full_formal_scale_is_preserved": all(int(row["decisions"]) == 28_311_552 for row in methods.values()),
        "G05_topk_k4_is_exhaustively_hard_action_equivalent": int(eq["grid_points"]) == 1_048_576 and int(eq["hard_disagreements"]) == 0 and eq["topk_action_sha256"] == eq["full_action_sha256"],
        "G06_topk_soft_approximation_is_not_hidden": bool(eq["soft_outputs_declared_approximate"]) and max(float(eq["q_llr_max_abs_error"]), float(eq["p_llr_max_abs_error"])) > 0.0,
        "G07_paired_static_contrast_falsifies_route_a_superiority": float(contrast["ci95_high"]) < 0.0 and not bool(contrast["route_a_superiority_passes_lcb_gt_zero"]),
        "G08_literature_rows_are_primary_and_forbid_cross_model_raw_ranking": len(report["literature_registry"]) >= 4 and all(row["primary_url"].startswith("https://") and row["numeric_cross_model_comparison_allowed"] is False for row in report["literature_registry"]),
        "G09_claim_registry_freezes_negative_result": claims == {"STATIC_GKP_ROUTE_A_SUPERIORITY": "FALSIFIED", "TOPK_K4_HARD_ACTION_EQUIVALENCE": "ESTABLISHED_PREBOARD", "GLOBAL_GKP_SOTA": "PROHIBITED", "PHYSICAL_BREAK_EVEN": "PROHIBITED"},
        "G10_cost_and_hardware_scope_are_explicit": (
            methods["topk_k4_static_map"]["cost"]["target_measured"] is False
            and methods["static_joint_map"]["cost"]["target_measured"] is False
            and methods["proposed_route_a"]["cost"]["board_measured"] is False
            and int(methods["topk_k4_static_map"]["cost"]["operation_storage_proxy"]["serial_cycle_upper_proxy"])
            < int(methods["static_joint_map"]["cost"]["operation_storage_proxy"]["serial_cycle_upper_proxy"])
            and int(methods["topk_k4_static_map"]["cost"]["operation_storage_proxy"]["retained_state_bits"])
            < int(methods["static_joint_map"]["cost"]["operation_storage_proxy"]["retained_state_bits"])
        ),
        "G11_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["detected"] == report["semantic_mutation_audit"]["count"] == 8,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    def attempt(name: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        mutate(candidate)
        candidate["semantic_mutation_audit"] = {"count": 8, "detected": 8, "cases": []}
        rows.append({"case": name, "rejected": not all(evaluate_gates(candidate).values())})
    attempt("parent_unverified", lambda x: x["parent_audit"].update(formal_report_verified=False))
    attempt("oracle_deployable", lambda x: next(row for row in x["method_table"] if row["method_id"] == "hidden_state_oracle").update(deployable=True))
    attempt("drop_pauli_metric", lambda x: next(row for row in x["method_table"] if row["method_id"] == "proposed_route_a").pop("p_Y"))
    attempt("shrink_trace", lambda x: next(row for row in x["method_table"] if row["method_id"] == "static_joint_map").update(decisions=1024))
    attempt("inject_grid_disagreement", lambda x: x["topk_full_exhaustive_equivalence"].update(hard_disagreements=1))
    attempt("hide_soft_error", lambda x: x["topk_full_exhaustive_equivalence"].update(q_llr_max_abs_error=0.0, p_llr_max_abs_error=0.0))
    attempt("promote_route_a", lambda x: x["claim_registry"][0].update(state="ESTABLISHED"))
    attempt("claim_measured_topk", lambda x: next(row for row in x["method_table"] if row["method_id"] == "topk_k4_static_map")["cost"].update(target_measured=True))
    return {"count": len(rows), "detected": sum(row["rejected"] for row in rows), "cases": rows}


def build_report() -> dict[str, Any]:
    parent = _load(PARENT)
    smooth.verify_report(parent)
    equivalence = _exhaustive_topk_equivalence()
    table = _method_table(parent, equivalence)
    contrast = _paired_static_contrast(parent)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same T6.7.1 smooth formal simulator/protocol and complete 10-bit deployable syndrome domain; not cross-paper raw LER ranking",
        "parent_audit": {
            "formal_path": _relative(PARENT),
            "formal_sha256": _sha256(PARENT),
            "topk_validation_path": _relative(TOPK_PARENT),
            "topk_validation_sha256": _sha256(TOPK_PARENT),
            "formal_report_verified": True,
            "formal_trajectories": len(parent["trajectory_results"]),
            "formal_decisions_per_method": 28_311_552,
        },
        "implementation_binding": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "literature_registry": _literature_registry(),
        "topk_full_exhaustive_equivalence": equivalence,
        "method_table": table,
        "paired_static_contrast": contrast,
        "oracle_gap": parent["analysis"]["oracle_gap_closure"],
        "claim_registry": [
            {"claim_id": "STATIC_GKP_ROUTE_A_SUPERIORITY", "state": "FALSIFIED", "reason": "paired static-minus-Route-A CI is strictly below zero"},
            {"claim_id": "TOPK_K4_HARD_ACTION_EQUIVALENCE", "state": "ESTABLISHED_PREBOARD", "reason": "zero disagreement on all 1024x1024 deployable syndrome pairs"},
            {"claim_id": "GLOBAL_GKP_SOTA", "state": "PROHIBITED", "reason": "literature models and physical protocols are not numerically commensurate"},
            {"claim_id": "PHYSICAL_BREAK_EVEN", "state": "PROHIBITED", "reason": "this is syndrome-level simulation, not a measured logical memory"},
        ],
        "allowed_wording": [
            "On the frozen same-model smooth formal trace, Route-A is significantly worse in average LER than frozen full static MAP.",
            "For the frozen static model, K=4 and full MAP have identical hard actions on the complete 10-bit syndrome domain, with lower deterministic operation/storage proxy but approximate soft outputs.",
        ],
        "forbidden_wording": [
            "Route-A outperforms static GKP decoding.",
            "The project exceeds literature GKP LER or physical break-even.",
            "K=4 is universally exact for all noise models or already measured on FPGA.",
        ],
    }
    report["semantic_mutation_audit"] = {"count": 8, "detected": 8, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_STATIC_GKP_SAME_MODEL_LANE"
    return report


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in report["method_table"]:
        rows.append({
            "row_type": "method_summary", "key": method["method_id"],
            "family": "aggregate", "value": method["average_ler_equal_family_seed"],
            "detail": json.dumps({key: method[key] for key in ("p_L", "p_X", "p_Y", "p_Z", "p95_window_ler", "worst_window_ler")}, sort_keys=True),
        })
        for family, value in method["family_ler"].items():
            rows.append({"row_type": "family_ler", "key": method["method_id"], "family": family, "value": value, "detail": "same formal trace"})
    for chunk in report["topk_full_exhaustive_equivalence"]["chunks"]:
        rows.append({"row_type": "grid_chunk", "key": f"q{chunk['q_start']}:{chunk['q_stop']}", "family": "complete_adc_domain", "value": chunk["hard_disagreements"], "detail": json.dumps(chunk, sort_keys=True)})
    for gate_name, value in report["gates"].items():
        rows.append({"row_type": "gate", "key": gate_name, "family": "", "value": str(bool(value)).lower(), "detail": ""})
    return rows


def write_report(report: dict[str, Any], artifact: Path, source_data: Path) -> None:
    rows = _source_rows(report)
    source_data.parent.mkdir(parents=True, exist_ok=True)
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_type", "key", "family", "value", "detail"))
        writer.writeheader(); writer.writerows(rows)
    report["output_source_data_binding"] = {"path": _relative(source_data), "sha256": _sha256(source_data), "row_count": len(rows)}
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if report.get("gates") != gates or not all(gates.values()) or report.get("verdict") != VERDICT:
        raise ValueError("T6.8.1 gates or verdict do not recompute")
    parent = report["parent_audit"]
    for path_key, sha_key in (("formal_path", "formal_sha256"), ("topk_validation_path", "topk_validation_sha256")):
        if _sha256(ROOT / parent[path_key]) != parent[sha_key]:
            raise ValueError(f"T6.8.1 parent drifted: {parent[path_key]}")
    implementation = report["implementation_binding"]
    if _sha256(ROOT / implementation["path"]) != implementation["sha256"]:
        raise ValueError("T6.8.1 implementation drifted")
    output = report.get("output_source_data_binding")
    if not output:
        raise ValueError("T6.8.1 Source Data is unbound")
    path = ROOT / output["path"]
    if _sha256(path) != output["sha256"] or _csv_rows(path) != int(output["row_count"]):
        raise ValueError("T6.8.1 Source Data drifted")
    if not all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]):
        raise ValueError("T6.8.1 mutation audit is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.artifact, args.source_data)
    verify_report(_load(args.artifact))
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "contrast": report["paired_static_contrast"], "grid": {key: report["topk_full_exhaustive_equivalence"][key] for key in ("grid_points", "hard_disagreements", "q_llr_max_abs_error", "p_llr_max_abs_error")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
