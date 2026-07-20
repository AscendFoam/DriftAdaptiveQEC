"""T7.1.4 evidence-bounded Supplement figure contract."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import held_out_ood_validation as ood_validation
from cnn_fpga.benchmark import logical_channel_reconstruction as logical_channel
from cnn_fpga.benchmark import main_result_figure_contract as main_results
from cnn_fpga.benchmark import qec_channel_recovery_bound as recovery_bound
from cnn_fpga.benchmark import route_a_smooth_formal as smooth_formal
from cnn_fpga.benchmark import route_a_tail_formal as tail_formal
from cnn_fpga.benchmark import secondary_evidence_integrity_gate as secondary_gate


# NumPy < 2.0 exposes the same composite trapezoid rule as ``trapz``.
# Parent reports were produced in an environment where ``trapezoid`` exists;
# keep live validation available in the repository default environment.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.1.4"
SCHEMA_VERSION = "t7.1.4-supplement-figure-contract-v1"
VERDICT = "PASS_SUPPLEMENT_FIGURE_CONTRACT_RESTRICTED_NONRANKING"

DEFAULT_REPORT = ROOT / "docs/t7_1_4_supplement_figure_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_1_4_supplement_figure_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/supplement_figures_contract.md"
FIGURE_DIR = ROOT / "docs/figures/t7_1_4_supplement_figures"
DEFAULT_MANIFEST = FIGURE_DIR / "figure_manifest.json"

SOURCES = {
    "gradient_report": ROOT / "docs/t2_3_5_feedback_grape_gradient_validation.json",
    "gradient_code": ROOT / "physics/feedback_grape_gradient.py",
    "feasibility_report": ROOT / "docs/t2_3_6_differentiable_sbs_feasibility.json",
    "feasibility_source": ROOT / "docs/t2_3_6_differentiable_sbs_feasibility.csv",
    "feasibility_code": ROOT / "physics/differentiable_sbs_feasibility.py",
    "noise_report": ROOT / "docs/t2_3_8_noise_transfer_validation.json",
    "noise_code": ROOT / "physics/noise_transfer_surrogate.py",
    "petz_report": ROOT / "docs/t5_3_5_qec_channel_recovery_bound.json",
    "petz_source": ROOT / "docs/t5_3_5_qec_channel_recovery_bound_source_data.csv",
    "petz_code": ROOT / "cnn_fpga/benchmark/qec_channel_recovery_bound.py",
    "topk_report": ROOT / "docs/t3_1_5_topk_map_validation.json",
    "topk_source": ROOT / "docs/t3_1_5_topk_map_source_data.csv",
    "topk_code": ROOT / "cnn_fpga/benchmark/topk_lattice_coset_map.py",
    "channel_report": ROOT / "docs/t5_3_1_logical_channel_reconstruction.json",
    "channel_source": ROOT / "docs/t5_3_1_logical_channel_source_data.csv",
    "channel_code": ROOT / "cnn_fpga/benchmark/logical_channel_reconstruction.py",
    "smooth_report": ROOT / "docs/t6_7_1_smooth_formal_matrix.json",
    "smooth_source": ROOT / "docs/t6_7_1_smooth_formal_matrix_source_data.csv",
    "smooth_code": ROOT / "cnn_fpga/benchmark/route_a_smooth_formal.py",
    "ood_report": ROOT / "docs/t5_4_1_held_out_ood_validation.json",
    "ood_source": ROOT / "docs/t5_4_1_held_out_ood_source_data.csv",
    "ood_code": ROOT / "cnn_fpga/benchmark/held_out_ood_validation.py",
    "fixed_report": ROOT / "docs/t2_4_3_fixed_point_validation.json",
    "fixed_source": ROOT / "docs/t2_4_3_precision_resource_ler.csv",
    "fixed_code": ROOT / "cnn_fpga/runtime/fixed_point_chain.py",
    "fast_fixed_report": ROOT / "docs/t4_2_4_fast_path_fixed_point_validation.json",
    "fast_fixed_source": ROOT / "docs/t4_2_4_fast_path_fixed_point_ler.csv",
    "fast_fixed_code": ROOT / "cnn_fpga/benchmark/fast_path_fixed_point_validation.py",
    "tail_report": ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json",
    "tail_source": ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix_source_data.csv",
    "tail_code": ROOT / "cnn_fpga/benchmark/route_a_tail_formal.py",
    "main_result_report": ROOT / "docs/t7_1_3_main_result_figure_contract.json",
    "main_result_manifest": ROOT / "docs/figures/t7_1_3_main_figures/figure_manifest.json",
    "main_result_code": ROOT / "cnn_fpga/benchmark/main_result_figure_contract.py",
    "secondary_report": ROOT / "docs/t6_19_3_secondary_evidence_integrity.json",
    "secondary_source": ROOT / "docs/t6_19_3_secondary_evidence_integrity_source_data.csv",
    "secondary_code": ROOT / "cnn_fpga/benchmark/secondary_evidence_integrity_gate.py",
    "secondary_svg": ROOT / "docs/figures/t6_19_3_secondary_comparison_atlas.svg",
    "secondary_pdf": ROOT / "docs/figures/t6_19_3_secondary_comparison_atlas.pdf",
    "secondary_png": ROOT / "docs/figures/t6_19_3_secondary_comparison_atlas.png",
    "secondary_tiff": ROOT / "docs/figures/t6_19_3_secondary_comparison_atlas.tiff",
    "implementation": Path(__file__).resolve(),
}

GENERATED_OUTPUTS = tuple(
    f"supplement_s{index}_{stem}.{suffix}"
    for index, stem in (
        (1, "physics_validity"),
        (2, "bounds_maps_channel"),
        (3, "seeds_and_ood"),
        (4, "fixed_point_and_failures"),
    )
    for suffix in ("svg", "pdf", "png", "tiff")
)
LINKED_OUTPUTS = {
    "supplement_s5_phase6c_atlas.svg": "secondary_svg",
    "supplement_s5_phase6c_atlas.pdf": "secondary_pdf",
    "supplement_s5_phase6c_atlas.png": "secondary_png",
    "supplement_s5_phase6c_atlas.tiff": "secondary_tiff",
}
ALL_OUTPUTS = GENERATED_OUTPUTS + tuple(LINKED_OUTPUTS)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _simple_pass(report: Mapping[str, Any], *, passed_key: str | None = None) -> bool:
    status = report.get("status") == "PASS" or (passed_key is not None and report.get(passed_key) is True)
    checks = report.get("checks")
    gates = report.get("gate_summary", {}).get("gates") if isinstance(report.get("gate_summary"), Mapping) else None
    values = checks.values() if isinstance(checks, Mapping) else gates.values() if isinstance(gates, Mapping) else ()
    return bool(status and values and all(value is True for value in values))


def _parent_verification() -> dict[str, bool]:
    reports = {key.removesuffix("_report"): _load(path) for key, path in SOURCES.items() if key.endswith("_report")}
    calls = {
        "gradient": lambda: _simple_pass(reports["gradient"]),
        "feasibility": lambda: _simple_pass(reports["feasibility"]),
        "noise_transfer": lambda: _simple_pass(reports["noise"], passed_key="passed"),
        "petz_bound": lambda: all(recovery_bound.validate_artifact_payload(reports["petz"]).values()),
        "topk": lambda: _simple_pass(reports["topk"]),
        "six_pauli_channel": lambda: all(logical_channel.validate_artifact_payload(reports["channel"]).values()),
        "smooth_formal": lambda: smooth_formal.verify_report(reports["smooth"]) is not False,
        "held_out_ood": lambda: not ood_validation.validate_artifact(reports["ood"]),
        "fixed_point_oat": lambda: reports["fixed"].get("status") == "PASS" and all(reports["fixed"].get("gates", {}).values()),
        "fast_path_fixed_point": lambda: _simple_pass(reports["fast_fixed"]),
        "tail_formal": lambda: tail_formal.verify_report(reports["tail"]) is not False,
        "main_result_boundary": lambda: all(main_results.verify_report(path=SOURCES["main_result_report"]).values()) and all(main_results.verify_bundle(SOURCES["main_result_manifest"]).values()),
        "secondary_integrity": lambda: secondary_gate.verify_report(report_path=SOURCES["secondary_report"]) is not False,
    }
    result: dict[str, bool] = {}
    for key, call in calls.items():
        try:
            result[key] = bool(call())
        except Exception:
            result[key] = False
    return result


def _record(
    category: str, figure: str, panel: str, record_id: str, metric: str,
    method: str, family: str, value: Any, unit: str, status: str,
    evidence_layer: str, source_ids: Sequence[str], selector: str,
    lower: Any = None, upper: Any = None, metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "category": category, "figure": figure, "panel": panel, "record_id": record_id,
        "metric": metric, "method": method, "family": family, "value": value,
        "lower": lower, "upper": upper, "unit": unit, "status": status,
        "evidence_layer": evidence_layer, "source_ids": list(source_ids),
        "selector": selector, "metadata": dict(metadata or {}),
    }


def _smooth_seed_records(smooth: Mapping[str, Any]) -> list[dict[str, Any]]:
    totals: dict[tuple[int, str], list[int]] = {}
    for trajectory in smooth["trajectory_results"]:
        seed = int(trajectory["seed"])
        for method, windows in trajectory["method_window_pauli_counts_class_order_I_Z_X_Y"].items():
            accumulator = totals.setdefault((seed, method), [0, 0])
            for counts in windows:
                accumulator[0] += sum(int(value) for value in counts[1:])
                accumulator[1] += sum(int(value) for value in counts)
    return [
        _record(
            "all_seed_distribution", "Supplement S3", "a", f"seed_{seed}_{method}",
            "per_seed_equal_trace_ler", method, "smooth_formal_all_families", errors / denominator,
            "probability", "ALL_FORMAL_SEEDS", "PROJECT_NATIVE_SIMULATION",
            ("smooth_report", "smooth_source", "smooth_code"),
            f"trajectory_results[*].method_window_pauli_counts[{seed},{method}]",
            metadata={"seed": seed, "errors": errors, "denominator": denominator},
        )
        for (seed, method), (errors, denominator) in sorted(totals.items())
    ]


def _build_records() -> list[dict[str, Any]]:
    gradient = _load(SOURCES["gradient_report"])
    feasibility = _load(SOURCES["feasibility_report"])
    noise = _load(SOURCES["noise_report"])
    petz = _load(SOURCES["petz_report"])
    topk = _load(SOURCES["topk_report"])
    channel = _load(SOURCES["channel_report"])
    smooth = _load(SOURCES["smooth_report"])
    ood = _load(SOURCES["ood_report"])
    fixed = _load(SOURCES["fixed_report"])
    fast_fixed = _load(SOURCES["fast_fixed_report"])
    tail = _load(SOURCES["tail_report"])
    main = _load(SOURCES["main_result_report"])
    secondary = _load(SOURCES["secondary_report"])
    records: list[dict[str, Any]] = []

    for index, row in enumerate(gradient["finite_difference_step_sweep"]):
        records.append(_record(
            "gradient", "Supplement S1", "a", f"gradient_step_{index}", "relative_l2_error",
            "Feedback-GRAPE exact gradient", "finite_difference_step_sweep", row["relative_l2_error"], "relative error",
            "NUMERICAL_VALIDATION", "PROJECT_NATIVE_SIMULATION", ("gradient_report", "gradient_code"),
            f"finite_difference_step_sweep[{index}]", metadata={"step": row["step"], "reward_path_error": row["reward_path_relative_l2_error"], "score_path_error": row["score_path_relative_l2_error"]},
        ))
    for row in feasibility["points"]:
        records.append(_record(
            "cutoff_feasibility", "Supplement S1", "b", f"feas_{row['point_id']}", "runtime_median_seconds",
            "finite-cutoff differentiable SBS", row["device"], row.get("runtime_median_seconds"), "seconds",
            row["status"].upper(), "HOST_RESOURCE_MEASUREMENT", ("feasibility_report", "feasibility_source", "feasibility_code"),
            f"points[{row['point_id']}]", metadata={key: row.get(key) for key in ("cutoff", "batch_size", "full_cycles", "observed_memory_fraction", "within_runtime_budget", "within_memory_budget", "failure_kind")},
        ))
    for row in noise["squeezing_sweep"]:
        records.append(_record(
            "noise_transfer_domain", "Supplement S1", "c", f"noise_sweep_{row['squeezing_db']:g}db", "odd_alias_probability",
            "Heisenberg noise-transfer surrogate", row["validity"], row["odd_alias_probability"], "probability",
            "VALID_DOMAIN" if row["validity"] == "localized" else "FAILURE_DOMAIN", "ANALYTIC_SURROGATE",
            ("noise_report", "noise_code"), f"squeezing_sweep[{row['squeezing_db']}dB]",
            metadata={"squeezing_db": row["squeezing_db"], "central_probability": row["central_probability"], "clipping_ratio": row["clipping_ratio"], "decision_variance": row["decision_variance"]},
        ))
    for row in noise["fock_alignment"]:
        records.append(_record(
            "noise_transfer_alignment", "Supplement S1", "d", f"noise_align_{row['squeezing_db']:g}db", "proxy_to_direct_relative_error",
            "noise-transfer versus direct/Fock", "four_logical_states", row["maximum_proxy_to_direct_relative_error"], "relative error",
            "VALID_DOMAIN" if row["squeezing_db"] >= 10 else "FAILURE_DOMAIN", "CROSS_MODEL_VALIDATION",
            ("noise_report", "noise_code"), f"fock_alignment[{row['squeezing_db']}dB]",
            metadata={"squeezing_db": row["squeezing_db"], "fock_to_direct_relative_error": row["maximum_fock_to_direct_relative_error"], "state_relative_spread": row["direct_state_relative_spread"]},
        ))

    for row in petz["small_sdp_validation"]:
        records.append(_record(
            "petz_small_sdp", "Supplement S2", "a", f"petz_small_{row['row_id']}", "petz_fidelity",
            "Petz/transpose recovery", row["noise_profile"], row["petz"]["petz_fidelity"], "entanglement fidelity",
            "NONDEPLOYABLE_BOUND", "CHANNEL_RECOVERY_BOUND", ("petz_report", "petz_source", "petz_code"),
            f"small_sdp_validation[{row['row_id']}]", lower=row["sdp"]["intersection_certified_lower"], upper=row["sdp"]["intersection_certified_upper"],
            metadata={"cutoff": row["cutoff"], "certificate_width": row["sdp"]["intersection_width"], "petz_to_sdp_lower_gap": row["sdp"]["intersection_certified_lower"] - row["petz"]["petz_fidelity"]},
        ))
    for row in petz["extended_cutoff_scan"]:
        records.append(_record(
            "petz_cutoff_extension", "Supplement S2", "b", f"petz_ext_{row['row_id']}", "petz_infidelity",
            "Petz/transpose recovery", row["noise_profile"], row["petz"]["petz_infidelity"], "infidelity",
            "NONDEPLOYABLE_BOUND_NO_LARGE_CUTOFF_SDP", "CHANNEL_RECOVERY_BOUND", ("petz_report", "petz_source", "petz_code"),
            f"extended_cutoff_scan[{row['row_id']}]", metadata={"cutoff": row["cutoff"], "sdp": row["sdp"], "theorem_upper": row["petz"]["theorem_optimal_upper"]},
        ))
    for scenario in topk["scenarios"]:
        scenario_id = scenario["scenario"]["scenario_id"]
        for row in scenario["sweep"]:
            records.append(_record(
                "topk_pareto", "Supplement S2", "c", f"topk_{scenario_id}_k{row['K']}", "axis_llr_p99_abs_error",
                "top-K lattice-coset MAP", scenario_id, row["axis_llr_p99_abs_error"], "absolute LLR",
                "SOFTWARE_COST_PROXY_NOT_SYNTHESIS", "PROJECT_NATIVE_SIMULATION", ("topk_report", "topk_source", "topk_code"),
                f"scenarios[{scenario_id}].sweep[K={row['K']}]", metadata={"K": row["K"], "convergence_K": scenario["convergence_K"], "decision_disagreement_rate": row["decision_disagreement_rate"], "retained_state_bits": row["cost"]["retained_state_bits"], "serial_cycle_upper_proxy": row["cost"]["serial_cycle_upper_proxy"], "target_measured": row["cost"]["target_measured"]},
            ))
    for mode in ("qec_off", "qec_on"):
        lane = channel["lanes"][f"cutoff40:high:{mode}"]
        for cycle_index, cycle in enumerate(lane["cycles"]):
            for state_index, state in enumerate(lane["state_labels"]):
                records.append(_record(
                    "six_pauli_states", "Supplement S2", "d", f"pauli_{mode}_{cycle_index}_{state}", "code_subspace_survival",
                    mode, state, lane["survival"][cycle_index][state_index], "probability",
                    "FINITE_CUTOFF_CPTNI_NO_POSTSELECTION", "PROJECT_NATIVE_SIMULATION", ("channel_report", "channel_source", "channel_code"),
                    f"lanes[cutoff40:high:{mode}].survival[{cycle_index}][{state_index}]", metadata={"cycle": cycle, "time_us": lane["time_us"][cycle_index], "cutoff": 40, "noise_profile": "high"},
                ))

    records.extend(_smooth_seed_records(smooth))
    for aggregate in ood["drift_lane"]["scenario_aggregates"]:
        for method, values in aggregate["methods"].items():
            ci = values["error_rate_seed_cluster_ci"]
            records.append(_record(
                "ood_drift", "Supplement S3", "b", f"ood_drift_{aggregate['scenario_id']}_{method}", "logical_error_rate",
                method, aggregate["scenario_id"], ci["estimate"], "probability", "HELD_OUT_OOD_LANE_LOCAL",
                "PROJECT_NATIVE_SIMULATION", ("ood_report", "ood_source", "ood_code"), f"drift_lane.scenario_aggregates[{aggregate['scenario_id']}].methods[{method}]",
                lower=ci["ci_low"], upper=ci["ci_high"], metadata={"seed_clusters": aggregate["seed_clusters"], "exceeds_parent_envelope": aggregate["exceeds_parent_envelope"]},
            ))
    for aggregate in ood["measurement_confusion_lane"]["scenario_aggregates"]:
        ci = aggregate["misclassification_rate_seed_cluster_ci"]
        records.append(_record(
            "ood_measurement", "Supplement S3", "c", f"ood_confusion_{aggregate['scenario_id']}", "misclassification_rate",
            "observed measurement path", aggregate["scenario_id"], ci["estimate"], "probability", "HELD_OUT_OOD_LANE_LOCAL",
            "PROJECT_NATIVE_SIMULATION", ("ood_report", "ood_source", "ood_code"), f"measurement_confusion_lane.scenario_aggregates[{aggregate['scenario_id']}]",
            lower=ci["ci_low"], upper=ci["ci_high"], metadata={"target_g_to_e": aggregate["target_g_to_e"], "target_e_to_g": aggregate["target_e_to_g"]},
        ))
    for aggregate in ood["leakage_rate_lane"]["rate_aggregates"]:
        ci = aggregate["unsafe_declared_available_fraction_seed_cluster_ci"]
        rate = aggregate["intervention_rate"]
        records.append(_record(
            "ood_leakage", "Supplement S3", "c", f"ood_leakage_{rate}", "unsafe_declared_available_fraction",
            "leakage/reset FSM", "intervention_rate", ci["estimate"], "probability", "HELD_OUT_OOD_LANE_LOCAL",
            "PROJECT_NATIVE_SIMULATION", ("ood_report", "ood_source", "ood_code"), f"leakage_rate_lane.rate_aggregates[{rate}]",
            lower=ci["ci_low"], upper=ci["ci_high"], metadata={"intervention_rate": rate, "hidden_occupancy": aggregate["hidden_leakage_occupancy_seed_cluster_ci"]["estimate"], "p95_hidden_run": aggregate["p95_hidden_leakage_run_steps_seed_cluster_ci"]["estimate"]},
        ))
    for aggregate in ood["communication_lane"]["scenario_aggregates"]:
        for metric in ("logical_error_rate", "end_to_end_control_availability"):
            values = aggregate[metric]
            records.append(_record(
                "ood_communication", "Supplement S3", "d", f"ood_comm_{aggregate['scenario']}_{metric}", metric,
                "scheduler/transport abstraction", aggregate["scenario"], values["mean"], "probability", "SOFTWARE_TIMING_MODEL_NOT_BOARD",
                "PROJECT_NATIVE_SIMULATION", ("ood_report", "ood_source", "ood_code"), f"communication_lane.scenario_aggregates[{aggregate['scenario']}].{metric}",
                lower=values["min"], upper=values["max"], metadata={"event_totals": aggregate["event_totals"], "target_hardware_measured": False},
            ))

    for aggregate in fixed["aggregates"]:
        records.append(_record(
            "fixed_point_oat", "Supplement S4", "a", f"fixed_oat_{aggregate['profile_id']}_{aggregate['bank_fault_mode']}", "paired_ler_minus_float",
            aggregate["profile_id"], aggregate["curve_axis"], aggregate["paired_ler_minus_float"]["mean"], "absolute LER",
            "REPRESENTATION_PROXY_NOT_SYNTHESIS", "FIXED_POINT_SIMULATION", ("fixed_report", "fixed_source", "fixed_code"),
            f"aggregates[{aggregate['profile_id']},{aggregate['bank_fault_mode']}]", lower=aggregate["paired_ler_minus_float"]["ci_low"], upper=aggregate["paired_ler_minus_float"]["ci_high"],
            metadata={"axis_value": aggregate["axis_value"], "storage_bits": aggregate["resource_proxy"]["total_dual_bank_storage_bits"], "disagreement": aggregate["prediction_disagreement_vs_float_mean"], "fault_events": aggregate["fault_events_total"], "target_synthesis_measured": aggregate["resource_proxy"]["target_synthesis_measured"]},
        ))
    for profile_id, values in fast_fixed["ler_summary"].items():
        profile = fast_fixed["profile_configs"][profile_id]
        ci = values["paired_quantized_minus_float"]
        records.append(_record(
            "fixed_point_production", "Supplement S4", "b", f"fast_fixed_{profile_id}", "paired_quantized_minus_float_ler",
            profile_id, "production_fast_path", ci["mean"], "absolute LER", "BIT_ACCURATE_SOFTWARE_NOT_BOARD",
            "FIXED_POINT_SIMULATION", ("fast_fixed_report", "fast_fixed_source", "fast_fixed_code"), f"ler_summary[{profile_id}]",
            lower=ci["ci_low"], upper=ci["ci_high"], metadata={"action_disagreement": values["action_disagreement_mean"], "adc_bits": profile["adc_bits"], "address_bits": profile["address_bits"], "llr_word_bits": profile["llr_word_bits"], "latency_cycles": profile["pipeline_latency_cycles"], "ii": profile["initiation_interval_cycles"]},
        ))
    failures = [
        ("low_squeezing_surrogate", 3, "FAILURE_DOMAIN", "noise_report", "3 dB proxy error/clipping"),
        ("feasibility_memory_boundary", feasibility["summary"]["status_counts"].get("memory_exceeded", 0), "RESOURCE_BOUNDARY", "feasibility_report", "memory-exceeded points"),
        ("feasibility_runtime_boundary", feasibility["summary"]["status_counts"].get("runtime_exceeded", 0), "RESOURCE_BOUNDARY", "feasibility_report", "runtime-exceeded points"),
        ("topk_hardware_unmeasured", None, "NOT_SYNTHESIZED", "topk_report", "cost proxy only"),
        ("petz_teacher_student_gap", None, "INCOMPARABLE", "petz_report", "heterogeneous metric/horizon"),
        ("ood_system_robustness", None, ood["system_robustness_status"], "ood_report", "lane-local only"),
        ("route_a_broad_tail_gain", False, "NOT_ESTABLISHED", "tail_report", "safety/non-inferiority only"),
        ("v5_execution", None, "DROPPED", "main_result_report", "causal-headroom early stop"),
        ("physical_board", next(row["value"] for row in main["records"] if row["record_id"] == "f4e_board_null"), "BLOCKED_ALL_NULL", "main_result_report", "measured fields null"),
    ]
    for index, (name, value, status, source, detail) in enumerate(failures):
        records.append(_record(
            "failure_mode", "Supplement S4", "d", f"failure_{index}_{name}", "failure_or_boundary_state",
            name, "registered boundary", value, "state", status, "BOUNDARY_LEDGER", (source,), detail, metadata={"detail": detail},
        ))
    for lane_id in sorted({cell["lane_id"] for cell in secondary["cells"]}):
        cells = [cell for cell in secondary["cells"] if cell["lane_id"] == lane_id]
        records.append(_record(
            "phase6c_linked_atlas", "Supplement S5", lane_id, f"phase6c_{lane_id}", "linked_cell_count",
            "secondary comparison atlas", lane_id, len(cells), "cells", "INDEPENDENT_LANE_NONRANKING",
            "MIXED_DECLARED_EVIDENCE", ("secondary_report", "secondary_source", "secondary_code", "secondary_svg", "secondary_pdf", "secondary_png", "secondary_tiff"),
            f"cells[lane_id={lane_id}]", metadata={"task_signatures": sorted({cell["task_signature_id"] for cell in cells}), "value_states": sorted({cell["value_state"] for cell in cells}), "evidence_grades": sorted({cell["evidence_grade"] for cell in cells})},
        ))
    return records


def _write_source_data(records: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    fields = ["category", "figure", "panel", "record_id", "metric", "method", "family", "value_json", "lower_json", "upper_json", "unit", "status", "evidence_layer", "source_ids_json", "source_hashes_json", "selector", "metadata_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                **{key: row[key] for key in ("category", "figure", "panel", "record_id", "metric", "method", "family", "unit", "status", "evidence_layer", "selector")},
                "value_json": json.dumps(row["value"], ensure_ascii=False), "lower_json": json.dumps(row["lower"], ensure_ascii=False), "upper_json": json.dumps(row["upper"], ensure_ascii=False),
                "source_ids_json": json.dumps(row["source_ids"], separators=(",", ":")),
                "source_hashes_json": json.dumps({key: artifacts[key]["sha256"] for key in row["source_ids"]}, sort_keys=True, separators=(",", ":")),
                "metadata_json": json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            })
    temporary.replace(path)


def _render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "# T7.1.4 Supplement figure contract", "",
        "## 冻结原则", "",
        "- Supplement 不是第二个排行榜；每个 panel 保留自己的 protocol、metric、information set、evidence grade 和 failure boundary。",
        "- S1--S4 由 Python/matplotlib 生成 editable SVG/PDF、300-dpi PNG、600-dpi LZW-TIFF；S5 原样链接 T6.19.3 已审计六-lane atlas。",
        "- Petz/SDP 是 arbitrary terminal recovery bound，不是 decoder/controller；top-K 成本是软件确定性 proxy，不是综合结果。",
        "- 六 Pauli 曲线是 finite-cutoff CPTNI code-subspace survival，无 post-selection、无实验 tomography claim。", "",
        "## Figure map", "",
        "- S1：Feedback-GRAPE gradient、cutoff/batch/horizon feasibility、noise-transfer valid/failure domain。",
        "- S2：small-cutoff Petz/SDP certificate、cutoff extension、top-K accuracy--cost proxy、six Pauli eigenstates。",
        "- S3：全部 24 formal seed×7 methods、四条 held-out/OOD lane；system/device robustness 保持 NOT ESTABLISHED。",
        "- S4：46 个 fixed-point OAT/bank aggregate、4 个 production integer profiles 和 9 项 failure/boundary ledger。",
        "- S5：Phase 6C 六条独立 lane 的 206-cell atlas；不压缩为 global score 或跨 task winner。", "",
        "## 禁止提升", "",
        *[f"- {item}" for item in report["forbidden_promotions"]], "",
        "Source data are provided. All plotted values and linked-atlas boundaries are hash-bound to live artifacts.", "",
    ])


def _csv_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["record_id"] for row in csv.DictReader(handle)]


def evaluate_gates(report: Mapping[str, Any], check_live_files: bool = True) -> dict[str, bool]:
    records = {row["record_id"]: row for row in report["records"]}
    categories: dict[str, list[Mapping[str, Any]]] = {}
    for row in records.values():
        categories.setdefault(row["category"], []).append(row)
    artifacts = report["artifact_registry"]
    source_path = ROOT / report["source_data"]["path"]
    markdown_path = ROOT / report["markdown"]["path"]
    linked = report["linked_outputs"]
    return {
        "G01_all_parent_verifiers_pass_live": len(report["parent_verification"]) == 13 and all(report["parent_verification"].values()),
        "G02_all_sources_are_hash_live_and_records_bound": all((not check_live_files or _live(value)) for value in artifacts.values()) and all(row["source_ids"] and set(row["source_ids"]) <= set(artifacts) and row["selector"] for row in records.values()),
        "G03_five_figures_have_exact_nonranking_panel_contract": set(report["figures"]) == {f"Supplement S{i}" for i in range(1, 6)} and all(report["figures"][f"Supplement S{i}"]["width_mm"] == 183 for i in range(1, 5)) and report["figures"]["Supplement S5"]["mode"] == "LINKED_VERIFIED_ATLAS",
        "G04_records_are_unique_and_all_required_categories_present": len(records) == len(report["records"]) and set(categories) == set(report["required_categories"]) and all(categories.values()),
        "G05_gradient_and_cutoff_resource_boundaries_are_complete": len(categories["gradient"]) == 4 and len(categories["cutoff_feasibility"]) == 65 and sum(row["status"] == "MEMORY_EXCEEDED" for row in categories["cutoff_feasibility"]) == 1 and sum(row["status"] == "RUNTIME_EXCEEDED" for row in categories["cutoff_feasibility"]) == 1,
        "G06_noise_transfer_valid_and_failure_domains_are_both_visible": {row["metadata"]["squeezing_db"] for row in categories["noise_transfer_domain"] if row["status"] == "FAILURE_DOMAIN"} == {3.0, 5.0, 8.0} and {row["metadata"]["squeezing_db"] for row in categories["noise_transfer_domain"] if row["status"] == "VALID_DOMAIN"} == {10.0, 12.0} and len(categories["noise_transfer_alignment"]) == 3,
        "G07_petz_sdp_is_bound_only_and_gap_incomparable": len(categories["petz_small_sdp"]) == 15 and len(categories["petz_cutoff_extension"]) == 15 and all(row["status"].startswith("NONDEPLOYABLE_BOUND") for row in categories["petz_small_sdp"] + categories["petz_cutoff_extension"]) and report["result_boundary"]["petz_teacher_student_gap"] == "INCOMPARABLE",
        "G08_topk_has_all_scenarios_k_and_no_hardware_promotion": len(categories["topk_pareto"]) == 48 and {row["metadata"]["K"] for row in categories["topk_pareto"]} == {1, 2, 4, 8, 16, 32, 64, 128} and {row["metadata"]["convergence_K"] for row in categories["topk_pareto"]} <= {2, 3, 4} and all(row["metadata"]["target_measured"] is False for row in categories["topk_pareto"]),
        "G09_six_pauli_states_are_complete_finite_cutoff_no_postselection": len(categories["six_pauli_states"]) == 372 and {row["family"] for row in categories["six_pauli_states"]} == {"x_plus", "x_minus", "y_plus", "y_minus", "z_plus", "z_minus"} and {row["method"] for row in categories["six_pauli_states"]} == {"qec_off", "qec_on"} and all(row["metadata"]["cutoff"] == 40 for row in categories["six_pauli_states"]),
        "G10_all_formal_seed_distributions_are_present": len(categories["all_seed_distribution"]) == 168 and len({row["metadata"]["seed"] for row in categories["all_seed_distribution"]}) == 24 and len({row["method"] for row in categories["all_seed_distribution"]}) == 7,
        "G11_full_ood_lanes_and_negative_system_status_are_explicit": len(categories["ood_drift"]) == 18 and len(categories["ood_measurement"]) == 3 and len(categories["ood_leakage"]) == 3 and len(categories["ood_communication"]) == 8 and report["result_boundary"]["ood_system_robustness"] == "NOT_ESTABLISHED_LANE_LOCAL_ONLY" and report["result_boundary"]["ood_device_robustness"] == "NOT_ESTABLISHED_NO_TARGET_HARDWARE",
        "G12_fixed_point_keeps_proxy_software_and_hardware_layers_separate": len(categories["fixed_point_oat"]) == 46 and len(categories["fixed_point_production"]) == 4 and all(row["metadata"]["target_synthesis_measured"] is False for row in categories["fixed_point_oat"]) and report["result_boundary"]["fixed_point_board_measured"] is False,
        "G13_failure_ledger_keeps_all_negative_null_and_dropped_states": len(categories["failure_mode"]) == 9 and {row["status"] for row in categories["failure_mode"]} >= {"FAILURE_DOMAIN", "RESOURCE_BOUNDARY", "NOT_SYNTHESIZED", "INCOMPARABLE", "NOT_ESTABLISHED", "DROPPED", "BLOCKED_ALL_NULL"},
        "G14_phase6c_atlas_is_six_lane_linked_and_not_globally_ranked": len(categories["phase6c_linked_atlas"]) == 6 and sum(int(row["value"]) for row in categories["phase6c_linked_atlas"]) == 206 and set(linked) == set(LINKED_OUTPUTS) and all(linked[name] == artifacts[source] for name, source in LINKED_OUTPUTS.items()) and report["result_boundary"]["phase6c_global_ranking"] is False,
        "G15_forbidden_promotions_are_complete": report["forbidden_promotions"] == ["Supplement as a global leaderboard", "Petz/SDP bound as deployable decoder or controller", "top-K operation/storage proxy as synthesized hardware", "finite-cutoff or 40-cutoff as infinite-dimensional convergence", "lane-local OOD as system/device robustness", "CXXRTL/P&R estimate as board measurement", "V5 resurrection after causal-headroom early stop", "measured FPGA speed/power advantage"],
        "G16_source_markdown_and_export_contract_are_live": source_path.is_file() and len(_csv_ids(source_path)) == len(records) and set(_csv_ids(source_path)) == set(records) and markdown_path.is_file() and all(token in markdown_path.read_text(encoding="utf-8") for token in ("不是第二个排行榜", "arbitrary terminal recovery bound", "全部 24 formal seed", "system/device robustness 保持 NOT ESTABLISHED", "Source data are provided")) and report["export_contract"] == {"backend": "Python/matplotlib only", "width_mm": 183, "svg_text": "editable", "pdf_fonttype": 42, "tiff_dpi": 600, "png_dpi": 300, "generated_outputs": list(GENERATED_OUTPUTS), "linked_outputs": list(LINKED_OUTPUTS)} and (not check_live_files or _live(report["source_data"]) and _live(report["markdown"])),
        "G17_one_substantive_mutation_per_gate_fails_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 17 and len(report["semantic_mutation_audit"]["cases"]) == 17,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    def row(value: dict[str, Any], record_id: str) -> dict[str, Any]:
        return next(item for item in value["records"] if item["record_id"] == record_id)
    def attempt(name: str, gate: str, change: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 17, "detected": 17, "cases": []}
        change(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})
    attempt("parent_failure", "G01_all_parent_verifiers_pass_live", lambda x: x["parent_verification"].update(noise_transfer=False))
    attempt("empty_source", "G02_all_sources_are_hash_live_and_records_bound", lambda x: row(x, "gradient_step_0").update(source_ids=[]))
    attempt("drop_s5", "G03_five_figures_have_exact_nonranking_panel_contract", lambda x: x["figures"].pop("Supplement S5"))
    attempt("duplicate_record", "G04_records_are_unique_and_all_required_categories_present", lambda x: x["records"][-1].update(record_id=x["records"][0]["record_id"]))
    attempt("hide_memory_boundary", "G05_gradient_and_cutoff_resource_boundaries_are_complete", lambda x: next(item for item in x["records"] if item["status"] == "MEMORY_EXCEEDED").update(status="PASS"))
    attempt("promote_3db", "G06_noise_transfer_valid_and_failure_domains_are_both_visible", lambda x: row(x, "noise_sweep_3db").update(status="VALID_DOMAIN"))
    attempt("deploy_petz", "G07_petz_sdp_is_bound_only_and_gap_incomparable", lambda x: next(item for item in x["records"] if item["category"] == "petz_small_sdp").update(status="DEPLOYABLE"))
    attempt("measure_topk", "G08_topk_has_all_scenarios_k_and_no_hardware_promotion", lambda x: next(item for item in x["records"] if item["category"] == "topk_pareto")["metadata"].update(target_measured=True))
    attempt("remove_pauli_state", "G09_six_pauli_states_are_complete_finite_cutoff_no_postselection", lambda x: x["records"].remove(next(item for item in x["records"] if item["category"] == "six_pauli_states")))
    attempt("hide_seed", "G10_all_formal_seed_distributions_are_present", lambda x: x["records"].remove(next(item for item in x["records"] if item["category"] == "all_seed_distribution")))
    attempt("promote_ood", "G11_full_ood_lanes_and_negative_system_status_are_explicit", lambda x: x["result_boundary"].update(ood_system_robustness="ESTABLISHED"))
    attempt("promote_fixed_hardware", "G12_fixed_point_keeps_proxy_software_and_hardware_layers_separate", lambda x: x["result_boundary"].update(fixed_point_board_measured=True))
    attempt("remove_failure", "G13_failure_ledger_keeps_all_negative_null_and_dropped_states", lambda x: x["records"].remove(next(item for item in x["records"] if item["category"] == "failure_mode")))
    attempt("rank_phase6c", "G14_phase6c_atlas_is_six_lane_linked_and_not_globally_ranked", lambda x: x["result_boundary"].update(phase6c_global_ranking=True))
    attempt("remove_forbidden", "G15_forbidden_promotions_are_complete", lambda x: x.update(forbidden_promotions=[]))
    attempt("switch_backend", "G16_source_markdown_and_export_contract_are_live", lambda x: x["export_contract"].update(backend="R"))
    attempt("forge_mutations", "G17_one_substantive_mutation_per_gate_fails_closed", lambda x: x.update(semantic_mutation_audit={"count": 17, "detected": 16, "cases": []}))
    return {"count": len(cases), "detected": sum(item["rejected"] for item in cases), "cases": cases}


def build_report(source_data: Path = DEFAULT_SOURCE_DATA, markdown: Path = DEFAULT_MARKDOWN) -> dict[str, Any]:
    artifacts = {key: _binding(path) for key, path in SOURCES.items()}
    records = _build_records()
    _write_source_data(records, artifacts, source_data)
    report: dict[str, Any] = {
        "task_id": TASK_ID, "schema_version": SCHEMA_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figures": {
            "Supplement S1": {"width_mm": 183, "height_mm": 135, "panels": "abcd", "conclusion": "Gradient correctness and host feasibility are established only inside explicit cutoff/resource and noise-transfer validity domains."},
            "Supplement S2": {"width_mm": 183, "height_mm": 150, "panels": "abcd", "conclusion": "Bounds, approximation sensitivity and six-state channels remain distinct evidence objects with no deployability transfer."},
            "Supplement S3": {"width_mm": 183, "height_mm": 145, "panels": "abcd", "conclusion": "All seeds and all registered OOD lanes expose method dispersion and communication/fault degradation rather than a selected average."},
            "Supplement S4": {"width_mm": 183, "height_mm": 125, "panels": "abcd", "conclusion": "Fixed-point retention is software/representation evidence and the failure ledger retains every null, dropped and negative boundary."},
            "Supplement S5": {"mode": "LINKED_VERIFIED_ATLAS", "panels": 6, "conclusion": "Six Phase 6C lanes remain independently scoped and cannot be collapsed into a global ranking."},
        },
        "artifact_registry": artifacts, "parent_verification": _parent_verification(), "records": records,
        "required_categories": ["gradient", "cutoff_feasibility", "noise_transfer_domain", "noise_transfer_alignment", "petz_small_sdp", "petz_cutoff_extension", "topk_pareto", "six_pauli_states", "all_seed_distribution", "ood_drift", "ood_measurement", "ood_leakage", "ood_communication", "fixed_point_oat", "fixed_point_production", "failure_mode", "phase6c_linked_atlas"],
        "result_boundary": {"petz_teacher_student_gap": "INCOMPARABLE", "ood_system_robustness": "NOT_ESTABLISHED_LANE_LOCAL_ONLY", "ood_device_robustness": "NOT_ESTABLISHED_NO_TARGET_HARDWARE", "fixed_point_board_measured": False, "phase6c_global_ranking": False, "v5_status": "DROPPED", "physical_board_status": "BLOCKED_ALL_NULL"},
        "forbidden_promotions": ["Supplement as a global leaderboard", "Petz/SDP bound as deployable decoder or controller", "top-K operation/storage proxy as synthesized hardware", "finite-cutoff or 40-cutoff as infinite-dimensional convergence", "lane-local OOD as system/device robustness", "CXXRTL/P&R estimate as board measurement", "V5 resurrection after causal-headroom early stop", "measured FPGA speed/power advantage"],
        "export_contract": {"backend": "Python/matplotlib only", "width_mm": 183, "svg_text": "editable", "pdf_fonttype": 42, "tiff_dpi": 600, "png_dpi": 300, "generated_outputs": list(GENERATED_OUTPUTS), "linked_outputs": list(LINKED_OUTPUTS)},
        "linked_outputs": {name: artifacts[source] for name, source in LINKED_OUTPUTS.items()},
        "source_data": {**_binding(source_data), "rows": len(records)}, "markdown": {"path": _relative(markdown), "sha256": "", "bytes": 0},
        "semantic_mutation_audit": {"count": 17, "detected": 17, "cases": []}, "verdict": VERDICT,
    }
    _atomic_text(_render_markdown(report), markdown)
    report["markdown"] = _binding(markdown)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": [key for key, value in report["gates"].items() if not value]}
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_SUPPLEMENT_FIGURE_CONTRACT"
    keys = ("figures", "artifact_registry", "parent_verification", "records", "required_categories", "result_boundary", "forbidden_promotions", "export_contract", "linked_outputs", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")
    report["analysis_sha256"] = _canonical_sha256({key: report[key] for key in keys})
    return report


def verify_report(report: Mapping[str, Any] | None = None, path: Path = DEFAULT_REPORT) -> dict[str, bool]:
    value = dict(report) if report is not None else _load(path)
    gates = evaluate_gates(value)
    keys = ("figures", "artifact_registry", "parent_verification", "records", "required_categories", "result_boundary", "forbidden_promotions", "export_contract", "linked_outputs", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")
    expected_hash = _canonical_sha256({key: value[key] for key in keys})
    checks = {"identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION, "gates": value.get("gates") == gates and all(gates.values()), "verdict": value.get("verdict") == VERDICT, "analysis_hash": value.get("analysis_sha256") == expected_hash}
    if not all(checks.values()):
        raise ValueError(f"T7.1.4 contract verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def verify_bundle(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, bool]:
    manifest = _load(manifest_path)
    outputs = manifest.get("outputs", {})
    checks = {
        "identity": manifest.get("task_id") == TASK_ID and manifest.get("backend") == "Python/matplotlib only",
        "contract_live": manifest.get("contract") == _binding(DEFAULT_REPORT), "source_data_live": manifest.get("source_data") == _binding(DEFAULT_SOURCE_DATA),
        "outputs_exact": set(outputs) == set(ALL_OUTPUTS), "outputs_live": set(outputs) == set(ALL_OUTPUTS) and all(_live(value) for value in outputs.values()),
        "linked_atlas_exact": all(outputs[name] == _binding(SOURCES[source]) for name, source in LINKED_OUTPUTS.items()),
        "editable_svg": manifest.get("qa", {}).get("svg_text_nodes", 0) >= 120 and manifest.get("qa", {}).get("svg_path_text_promotion") is False,
        "raster_dimensions": all(value >= 2800 for value in manifest.get("qa", {}).get("tiff_min_dimension_px", {}).values()),
        "visual_contract": manifest.get("qa", {}).get("backend_exclusive") is True and manifest.get("qa", {}).get("manual_visual_qa") == "PASS",
    }
    if not all(checks.values()):
        raise ValueError(f"T7.1.4 bundle verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        verify_report(path=args.report)
        print(json.dumps({"verified": _relative(args.report), "verdict": VERDICT}, ensure_ascii=False))
        return 0
    report = build_report(args.source_data, args.markdown)
    _atomic_json(report, args.report)
    verify_report(report, args.report)
    print(json.dumps({"output": _relative(args.report), "records": len(report["records"]), "gates": report["gate_summary"], "verdict": report["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
