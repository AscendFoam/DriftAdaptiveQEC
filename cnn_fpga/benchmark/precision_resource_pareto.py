"""Build the T5.5.3 precision/resource/performance Pareto decision."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any, Sequence

from cnn_fpga.benchmark.target_device_synthesis import (
    CORE as CORE_RTL,
    CST,
    DEVICE,
    FAMILY,
    SDC,
    TARGET_MHZ,
    TOP as CORE_TOP,
    _read_tool_text,
    parse_nextpnr,
    parse_yosys_log,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
DEFAULT_BUILD = ROOT / ".tmp_t553_build"
DEFAULT_JSON = ROOT / "docs/t5_5_3_precision_resource_pareto.json"
DEFAULT_CSV = ROOT / "docs/t5_5_3_precision_resource_pareto_source_data.csv"
PARENTS = {
    "precision": ROOT / "docs/t4_2_4_fast_path_fixed_point_validation.json",
    "topk": ROOT / "docs/t3_1_5_topk_map_validation.json",
    "student_dimension": ROOT / "docs/t4_4_3_low_dimensional_student_validation.json",
    "student_gain": ROOT / "docs/t4_4_4_teacher_student_gain_retention.json",
    "base_target_synthesis": ROOT / "docs/t5_5_2_target_device_synthesis.json",
    "student_rtl_equivalence": ROOT / "docs/t5_5_3_student_rtl_equivalence.json",
}
STUDENT_RTL = ROOT / "cnn_fpga/rtl/low_dimensional_student_kernel.sv"
INTEGRATED_TOP = ROOT / "cnn_fpga/rtl/gkp_fast_path_student_synth_top.sv"
STUDENT_MANIFEST = ROOT / "cnn_fpga/rtl/generated/t5_5_3_student_memory_manifest.json"
PRECISION_IDS = (
    "low_p6_a4_q5_6", "medium_p8_a6_q7_10",
    "selected_p10_a8_q9_12", "dense_p12_a10_q10_14",
)
TOPK_VALUES = (1, 2, 4)
STATE_DIMENSIONS = (1, 2, 4)
PARALLELISM_VALUES = (1, 2, 4)
STUDENT_DEADLINE_US = 5.0
VERDICT = "SELECT_P10_A8_Q9_12_K4_REFERENCE_STATE4_SERIAL_DSP_POST_ROUTE_PASS_NOT_BOARD_MEASURED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _matches(binding: dict[str, Any]) -> bool:
    path = ROOT / binding["path"]
    return path.is_file() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _load_parent(name: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_gate_summary = payload.get("gate_summary") or {}
    passed = int(raw_gate_summary.get("passed", 0))
    if "total" in raw_gate_summary:
        total = int(raw_gate_summary["total"])
    else:
        failed = raw_gate_summary.get("failed", 0)
        failed_count = len(failed) if isinstance(failed, list) else int(failed)
        total = passed + failed_count
    return payload, {
        "name": name,
        "artifact": _binding(path),
        "task_id": payload.get("task_id"),
        "status": payload.get("status"),
        "gate_summary": {"passed": passed, "total": total},
    }


def _copy_tool_artifacts(build_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    docs = ROOT / "docs"
    synthesis_source = build_dir / "yosys_integrated.log"
    synthesis_dest = docs / "t5_5_3_yosys_integrated.log"
    synthesis_dest.write_text(_read_tool_text(synthesis_source), encoding="utf-8")
    artifacts = [_binding(synthesis_dest)]
    for seed in (1, 7, 19):
        tag = f"{seed:02d}"
        report_source = build_dir / f"nextpnr_seed{tag}_report.json"
        log_source = build_dir / f"nextpnr_seed{tag}.log"
        report_dest = docs / f"t5_5_3_nextpnr_seed{tag}_report.json"
        log_dest = docs / f"t5_5_3_nextpnr_seed{tag}_place_route.log"
        shutil.copyfile(report_source, report_dest)
        log_dest.write_text(_read_tool_text(log_source), encoding="utf-8")
        artifacts.extend((_binding(report_dest), _binding(log_dest)))
    return synthesis_dest, artifacts


def precision_axis(parent: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for profile_id in PRECISION_IDS:
        config = parent["profile_configs"][profile_id]
        quality = parent["ler_summary"][profile_id]
        exhaustive = parent["exhaustive_code_summary"][profile_id]
        disagreement = float(quality["action_disagreement_mean"])
        interval = quality["paired_quantized_minus_float"]
        pass_quality = (
            exhaustive["hard_action_mismatch_count"] == 0
            and disagreement <= 1.0e-4
            and max(abs(float(interval["ci_low"])), abs(float(interval["ci_high"]))) <= 1.0e-3
        )
        entries = 2 ** int(config["address_bits"]) + 1
        word_bits = int(config["llr_word_bits"])
        blocks_per_physical_memory = math.ceil(entries * word_bits / 18432)
        rows.append({
            "profile_id": profile_id,
            "adc_bits": int(config["adc_bits"]),
            "address_bits": int(config["address_bits"]),
            "llr_word_bits": word_bits,
            "quantized_ler_mean": float(quality["quantized_ler_mean"]),
            "float_ler_mean": float(quality["float_ler_mean"]),
            "paired_ler_delta_ci": [float(interval["ci_low"]), float(interval["ci_high"])],
            "action_disagreement_mean": disagreement,
            "hard_action_mismatch_count": int(exhaustive["hard_action_mismatch_count"]),
            "quality_pass": pass_quality,
            "estimated_bram_blocks_for_eight_mirrors": 8 * blocks_per_physical_memory,
            "resource_scope": "exact_memory_packing_only_other_resources_require_synthesis",
        })
    return rows


def topk_axis(parent: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for k in TOPK_VALUES:
        scenario_rows = []
        for scenario in parent["scenarios"]:
            sweep = next(row for row in scenario["sweep"] if int(row["K"]) == k)
            scenario_rows.append({
                "scenario": scenario["scenario"]["scenario_id"],
                "convergence_k": int(scenario["convergence_K"]),
                "topk_ler": float(sweep["topk_map_ler"]),
                "full_ler": float(sweep["full_map_ler"]),
                "decision_disagreement_rate": float(sweep["decision_disagreement_rate"]),
                "axis_llr_p99_abs_error": float(sweep["axis_llr_p99_abs_error"]),
            })
        rows.append({
            "k": k,
            "scenario_count": len(scenario_rows),
            "all_scenarios_converged": all(row["convergence_k"] <= k for row in scenario_rows),
            "maximum_absolute_ler_delta": max(abs(row["topk_ler"] - row["full_ler"]) for row in scenario_rows),
            "maximum_decision_disagreement_rate": max(row["decision_disagreement_rate"] for row in scenario_rows),
            "maximum_axis_llr_p99_abs_error": max(row["axis_llr_p99_abs_error"] for row in scenario_rows),
            "scenario_rows": scenario_rows,
            "hardware_role": "off_device_reference_only_not_present_in_integrated_rtl",
        })
    return rows


def student_axis(dimension_parent: dict[str, Any], gain_parent: dict[str, Any]) -> list[dict[str, Any]]:
    selected = dimension_parent["selection"]
    all_retentions = []
    for split in gain_parent["stochastic_retention"].values():
        for metric in split.values():
            all_retentions.append((float(metric["point_retention_fraction"]), float(metric["ci_95"][0])))
    minimum_point = min(value[0] for value in all_retentions)
    minimum_ci = min(value[1] for value in all_retentions)
    rows = []
    for dimension in STATE_DIMENSIONS:
        metrics = dimension_parent["candidate_metrics"][str(dimension)]
        resource = metrics["resource_profile"]
        is_selected = dimension == int(selected["selected_dimension"])
        rows.append({
            "dimension": dimension,
            "validation_mse": float(metrics["validation"]["mse"]),
            "evaluation_mse": float(metrics["evaluation"]["mse"]),
            "stored_scalars": int(resource["stored_trainable_scalars"]),
            "analytic_macs_per_healthy_step": int(resource["multiply_adds_per_healthy_step"]),
            "parent_dimension_eligible": dimension in [int(value) for value in selected["eligible_dimensions"]],
            "physical_gain_retention_point_minimum": minimum_point if is_selected else None,
            "physical_gain_retention_ci_lower_minimum": minimum_ci if is_selected else None,
            "fixed_rtl_equivalence_available": is_selected,
        })
    return rows


def parallelism_axis() -> list[dict[str, Any]]:
    return [
        {
            "multipliers": parallelism,
            "selected_state4_latency_cycles": math.ceil(4 / parallelism) * 16,
            "selected_state4_latency_us_at_27mhz": math.ceil(4 / parallelism) * 16 / TARGET_MHZ,
            "evidence_level": "actual_fixed_rtl" if parallelism == 1 else "operation_count_extrapolation_not_rtl",
        }
        for parallelism in PARALLELISM_VALUES
    ]


def _max_resources(routes: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    names = ("LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9", "ALU", "IOB")
    return {
        name: {
            "used": max(row["utilization"][name]["used"] for row in routes),
            "available": routes[0]["utilization"][name]["available"],
        }
        for name in names
    }


def candidate_grid(
    precisions: list[dict[str, Any]], topks: list[dict[str, Any]],
    students: list[dict[str, Any]], parallelisms: list[dict[str, Any]],
    actual_resources: dict[str, dict[str, int]], base_resources: dict[str, dict[str, int]],
    actual_fmax_minimum: float,
) -> list[dict[str, Any]]:
    rows = []
    student_lut_increment = actual_resources["LUT4"]["used"] - base_resources["LUT4"]["used"]
    student_dff_increment = actual_resources["DFF"]["used"] - base_resources["DFF"]["used"]
    for precision in precisions:
        for topk in topks:
            for student in students:
                for parallel in parallelisms:
                    actual = (
                        precision["profile_id"] == "selected_p10_a8_q9_12"
                        and student["dimension"] == 4
                        and parallel["multipliers"] == 1
                    )
                    dimension_scale = student["dimension"] / 4.0
                    multiplier_scale = parallel["multipliers"]
                    estimated = {
                        "LUT4": base_resources["LUT4"]["used"] + math.ceil(student_lut_increment * dimension_scale * (1 + 0.1 * (multiplier_scale - 1))),
                        "DFF": base_resources["DFF"]["used"] + math.ceil(student_dff_increment * dimension_scale * (1 + 0.05 * (multiplier_scale - 1))),
                        "BSRAM": precision["estimated_bram_blocks_for_eight_mirrors"],
                        "MULT18X18": base_resources["MULT18X18"]["used"] + multiplier_scale,
                        "MULT9X9": base_resources["MULT9X9"]["used"],
                    }
                    quality_pass = (
                        precision["quality_pass"]
                        and topk["all_scenarios_converged"]
                        and student["parent_dimension_eligible"]
                    )
                    deadline_pass = parallel["selected_state4_latency_us_at_27mhz"] <= STUDENT_DEADLINE_US
                    measured_resources = {
                        name: value["used"] for name, value in actual_resources.items()
                    } if actual else None
                    final_eligible = quality_pass and deadline_pass and actual
                    rows.append({
                        "candidate_id": f"{precision['profile_id']}__k{topk['k']}__d{student['dimension']}__p{parallel['multipliers']}",
                        "precision_id": precision["profile_id"],
                        "topk_k": topk["k"],
                        "student_dimension": student["dimension"],
                        "parallel_multipliers": parallel["multipliers"],
                        "precision_quality_pass": precision["quality_pass"],
                        "topk_quality_pass": topk["all_scenarios_converged"],
                        "student_quality_pass": student["parent_dimension_eligible"],
                        "deadline_pass": deadline_pass,
                        "student_latency_cycles": parallel["selected_state4_latency_cycles"] if student["dimension"] == 4 else math.ceil(student["dimension"] / parallel["multipliers"]) * 16,
                        "student_latency_us_at_27mhz": (math.ceil(student["dimension"] / parallel["multipliers"]) * 16) / TARGET_MHZ,
                        "resource_evidence_level": "actual_three_seed_integrated_post_route" if actual else "calibrated_estimate_not_synthesis",
                        "measured_resources": measured_resources,
                        "estimated_resources": estimated,
                        "measured_fmax_mhz_minimum": actual_fmax_minimum if actual else None,
                        "topk_hardware_resources_included": False,
                        "final_eligible": final_eligible,
                    })
    return rows


def evaluate_gates(report: dict[str, Any]) -> dict[str, bool]:
    parents_ok = all(
        row["status"] == "PASS"
        and row["gate_summary"]
        and row["gate_summary"]["passed"] == row["gate_summary"]["total"]
        and _matches(row["artifact"])
        for row in report["parents"]
    )
    bindings_ok = all(_matches(row) for row in report["source_bindings"])
    artifacts_ok = all(_matches(row) for row in report["durable_artifacts"])
    precisions = report["axes"]["precision"]
    topks = report["axes"]["topk"]
    students = report["axes"]["student_dimension"]
    parallels = report["axes"]["parallelism"]
    precision_ok = (
        [row["profile_id"] for row in precisions] == list(PRECISION_IDS)
        and [row["profile_id"] for row in precisions if row["quality_pass"]]
        == ["selected_p10_a8_q9_12", "dense_p12_a10_q10_14"]
    )
    topk_ok = (
        [row["k"] for row in topks] == list(TOPK_VALUES)
        and [row["k"] for row in topks if row["all_scenarios_converged"]] == [4]
        and all(row["hardware_role"].startswith("off_device") for row in topks)
    )
    student_ok = (
        [row["dimension"] for row in students] == list(STATE_DIMENSIONS)
        and [row["dimension"] for row in students if row["parent_dimension_eligible"]] == [4]
        and students[-1]["physical_gain_retention_ci_lower_minimum"] >= 0.90
    )
    parallel_ok = (
        [row["multipliers"] for row in parallels] == list(PARALLELISM_VALUES)
        and parallels[0]["evidence_level"] == "actual_fixed_rtl"
        and all(row["selected_state4_latency_us_at_27mhz"] <= STUDENT_DEADLINE_US for row in parallels)
    )
    synthesis = report["integrated_synthesis"]
    base = report["base_post_route_resources"]
    actual = report["integrated_post_route_resources"]
    student_survives = (
        synthesis["zero_structural_problems"]
        and synthesis["cell_counts"]["SDPX9B"] == 8
        and synthesis["cell_counts"]["MULT18X18"] == 2
        and actual["DFF"]["used"] > base["DFF"]["used"]
        and actual["LUT4"]["used"] > base["LUT4"]["used"]
        and actual["MULT18X18"]["used"] == base["MULT18X18"]["used"] + 1
    )
    routes = report["integrated_place_route"]
    route_ok = (
        [row["seed"] for row in routes] == [1, 7, 19]
        and all(row["route_status"] == "PASS" and row["timing_pass"] for row in routes)
        and min(row["achieved_fmax_mhz"] for row in routes) >= TARGET_MHZ
    )
    resources_fit = all(value["used"] <= value["available"] for value in actual.values())
    candidates = report["candidates"]
    grid_ok = len(candidates) == 108 and len({row["candidate_id"] for row in candidates}) == 108
    selected_rows = [row for row in candidates if row["final_eligible"]]
    selected = report["selection"]
    unique_selection = (
        len(selected_rows) == 1
        and selected_rows[0]["candidate_id"] == selected["candidate_id"]
        and selected["precision_id"] == "selected_p10_a8_q9_12"
        and selected["topk_k"] == 4
        and selected["student_dimension"] == 4
        and selected["parallel_multipliers"] == 1
    )
    evidence_levels = all(
        (row["measured_resources"] is not None) == (row["resource_evidence_level"] == "actual_three_seed_integrated_post_route")
        and row["topk_hardware_resources_included"] is False
        for row in candidates
    )
    latency_ok = (
        selected["student_latency_cycles"] == 64
        and abs(selected["student_latency_us_at_27mhz"] - 64 / TARGET_MHZ) < 1e-12
        and selected["student_latency_us_at_27mhz"] <= STUDENT_DEADLINE_US
    )
    boundary = report["evidence_boundary"]
    boundary_ok = (
        boundary["candidate_pareto_audit"] is True
        and boundary["integrated_target_post_route"] is True
        and boundary["student_cxxrtl_equivalence"] is True
        and boundary["online_topk_rtl"] is False
        and boundary["vendor_timing_signoff"] is False
        and boundary["board_measured"] is False
    )
    return {
        "all_six_parent_artifacts_are_hash_bound_and_pass": parents_ok,
        "all_rtl_runner_memory_and_constraint_sources_are_hash_bound": bindings_ok,
        "four_precision_profiles_are_complete_and_thresholded": precision_ok,
        "k4_is_the_smallest_six_scenario_topk_reference_and_stays_off_device": topk_ok,
        "four_state_student_is_the_only_parent_eligible_dimension": student_ok,
        "parallelism_grid_separates_actual_serial_rtl_from_extrapolations": parallel_ok,
        "integrated_synthesis_preserves_core_and_one_additional_student_dsp": student_survives,
        "three_integrated_place_route_seeds_pass_27mhz": route_ok,
        "selected_integrated_resources_fit_the_exact_target_capacity": resources_fit,
        "full_108_point_joint_grid_is_unique": grid_ok,
        "exactly_one_source_bound_candidate_is_final_eligible": unique_selection,
        "no_estimate_or_off_device_topk_is_mislabeled_as_measured_hardware": evidence_levels,
        "serial_student_meets_the_five_microsecond_model_deadline": latency_ok,
        "all_durable_synthesis_and_place_route_artifacts_match_hashes": artifacts_ok,
        "post_route_pareto_is_not_mislabeled_as_vendor_or_board_evidence": boundary_ok,
    }


def mutation_audit(report: dict[str, Any]) -> list[dict[str, Any]]:
    mutations = [
        ("break_parent_hash", lambda r: r["parents"][0]["artifact"].__setitem__("sha256", "0" * 64)),
        ("make_k2_pass", lambda r: r["axes"]["topk"][1].__setitem__("all_scenarios_converged", True)),
        ("make_dimension2_eligible", lambda r: r["axes"]["student_dimension"][1].__setitem__("parent_dimension_eligible", True)),
        ("drop_joint_candidate", lambda r: r["candidates"].pop()),
        ("claim_estimate_as_measured", lambda r: r["candidates"][0].__setitem__("measured_resources", {})),
        ("claim_online_topk_hardware", lambda r: r["candidates"][0].__setitem__("topk_hardware_resources_included", True)),
        ("erase_student_dsp", lambda r: r["integrated_post_route_resources"]["MULT18X18"].__setitem__("used", 1)),
        ("miss_27mhz", lambda r: r["integrated_place_route"][0].__setitem__("achieved_fmax_mhz", 26.0)),
        ("break_deadline", lambda r: r["selection"].__setitem__("student_latency_us_at_27mhz", 6.0)),
        ("claim_board_measurement", lambda r: r["evidence_boundary"].__setitem__("board_measured", True)),
    ]
    rows = []
    for name, mutate in mutations:
        candidate = copy.deepcopy(report)
        mutate(candidate)
        gates = evaluate_gates(candidate)
        rows.append({"mutation": name, "rejected": not all(gates.values()), "failed_gates": [key for key, value in gates.items() if not value]})
    return rows


def build_report(build_dir: Path) -> dict[str, Any]:
    parent_payloads: dict[str, dict[str, Any]] = {}
    parent_rows = []
    for name, path in PARENTS.items():
        payload, row = _load_parent(name, path)
        parent_payloads[name] = payload
        parent_rows.append(row)
    synthesis_path, artifacts = _copy_tool_artifacts(build_dir)
    routes = [
        parse_nextpnr(
            ROOT / f"docs/t5_5_3_nextpnr_seed{seed:02d}_report.json",
            ROOT / f"docs/t5_5_3_nextpnr_seed{seed:02d}_place_route.log",
            seed,
        )
        for seed in (1, 7, 19)
    ]
    precisions = precision_axis(parent_payloads["precision"])
    topks = topk_axis(parent_payloads["topk"])
    students = student_axis(parent_payloads["student_dimension"], parent_payloads["student_gain"])
    parallels = parallelism_axis()
    actual_resources = _max_resources(routes)
    base_routes = parent_payloads["base_target_synthesis"]["place_route"]
    base_resources = _max_resources(base_routes)
    fmax = [row["achieved_fmax_mhz"] for row in routes]
    candidates = candidate_grid(
        precisions, topks, students, parallels, actual_resources, base_resources, min(fmax)
    )
    selected = next(row for row in candidates if row["final_eligible"])
    source_bindings = [_binding(path) for path in (
        RUNNER, CORE_RTL, CORE_TOP, STUDENT_RTL, INTEGRATED_TOP,
        STUDENT_MANIFEST, SDC, CST,
    )]
    manifest = json.loads(STUDENT_MANIFEST.read_text(encoding="utf-8"))
    source_bindings.extend(_binding(ROOT / row["path"]) for row in manifest["files"])
    report: dict[str, Any] = {
        "schema_version": "t5.5.3-precision-resource-performance-pareto-v1",
        "task_id": "T5.5.3",
        "status": "PENDING",
        "verdict": VERDICT,
        "target": {"device": DEVICE, "family": FAMILY, "target_mhz": TARGET_MHZ, "student_deadline_us": STUDENT_DEADLINE_US},
        "parents": parent_rows,
        "source_bindings": source_bindings,
        "axes": {
            "precision": precisions,
            "topk": topks,
            "student_dimension": students,
            "parallelism": parallels,
        },
        "integrated_synthesis": parse_yosys_log(synthesis_path),
        "integrated_place_route": routes,
        "base_post_route_resources": base_resources,
        "integrated_post_route_resources": actual_resources,
        "integrated_fmax_mhz": {
            "minimum": min(fmax), "median": statistics.median(fmax), "maximum": max(fmax),
        },
        "increment_over_t5_5_2_maxima": {
            name: actual_resources[name]["used"] - base_resources[name]["used"]
            for name in actual_resources
        },
        "candidates": candidates,
        "selection": selected,
        "selection_rule": (
            "filter parent quality gates; require source-bound actual integrated synthesis/P&R; "
            "then choose the smallest measured multiplier count meeting the 5 us project-model deadline"
        ),
        "durable_artifacts": artifacts,
        "evidence_boundary": {
            "candidate_pareto_audit": True,
            "student_cxxrtl_equivalence": True,
            "integrated_target_post_route": True,
            "online_topk_rtl": False,
            "parallelism_two_or_four_rtl": False,
            "vendor_timing_signoff": False,
            "bitstream_generated": False,
            "board_measured": False,
            "transport_implemented": False,
            "quantum_hardware_measured": False,
        },
    }
    report["gates"] = evaluate_gates(report)
    report["mutation_audit"] = mutation_audit(report)
    report["gates"]["all_ten_shortcut_mutations_are_rejected"] = all(row["rejected"] for row in report["mutation_audit"])
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    return report


def write_outputs(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    fields = (
        "candidate_id", "precision_id", "topk_k", "student_dimension", "parallel_multipliers",
        "precision_quality_pass", "topk_quality_pass", "student_quality_pass", "deadline_pass",
        "student_latency_cycles", "student_latency_us_at_27mhz", "resource_evidence_level",
        "measured_lut4", "measured_dff", "measured_bsram", "measured_mult18", "measured_mult9",
        "estimated_lut4", "estimated_dff", "estimated_bsram", "estimated_mult18", "estimated_mult9",
        "measured_fmax_mhz_minimum", "topk_hardware_resources_included", "final_eligible",
    )
    rows = []
    for row in report["candidates"]:
        measured = row["measured_resources"] or {}
        estimated = row["estimated_resources"]
        rows.append({
            **{name: row[name] for name in fields[:12]},
            "measured_lut4": measured.get("LUT4"), "measured_dff": measured.get("DFF"),
            "measured_bsram": measured.get("BSRAM"), "measured_mult18": measured.get("MULT18X18"),
            "measured_mult9": measured.get("MULT9X9"),
            "estimated_lut4": estimated["LUT4"], "estimated_dff": estimated["DFF"],
            "estimated_bsram": estimated["BSRAM"], "estimated_mult18": estimated["MULT18X18"],
            "estimated_mult9": estimated["MULT9X9"],
            "measured_fmax_mhz_minimum": row["measured_fmax_mhz_minimum"],
            "topk_hardware_resources_included": row["topk_hardware_resources_included"],
            "final_eligible": row["final_eligible"],
        })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report["source_data"] = {"path": _relative(csv_path), "candidate_rows": len(rows)}
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    report = build_report(args.build_dir.resolve())
    write_outputs(report, args.output_json.resolve(), args.output_csv.resolve())
    print(json.dumps({
        "status": report["status"], "verdict": report["verdict"],
        "selection": report["selection"]["candidate_id"],
        "fmax_mhz": report["integrated_fmax_mhz"], "gates": report["gate_summary"],
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
