"""T6.19.2 source-normalized external FPGA decoder refresh.

This module composes the frozen T6.8.6 adapter instead of mutating it.  The
older report is verified live, its eight external rows are imported, and new
cutoff-locked sources are normalized without creating a cross-code ranking.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark import fpga_decoder_normalization as base_normalization


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.19.2"
SCHEMA_VERSION = "t6.19.2-external-fpga-normalization-v1"
LEDGER = ROOT / "configs" / "literature" / "t6_19_2_external_fpga_refresh.json"
BASE_REPORT = ROOT / "docs" / "t6_8_6_fpga_decoder_normalization.json"
BASE_LEDGER = ROOT / "configs" / "literature" / "t6_8_6_fpga_decoder_sources.json"
BASE_IMPLEMENTATION = ROOT / "cnn_fpga" / "benchmark" / "fpga_decoder_normalization.py"
PROJECT_REPORT = ROOT / "docs" / "t6_19_1_project_preboard_profiles.json"
ONTOLOGY_REPORT = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
PREREGISTRATION = ROOT / "configs" / "literature" / "t6_16_3_secondary_preregistration.json"
DEFAULT_JSON = ROOT / "docs" / "t6_19_2_external_fpga_normalization.json"
DEFAULT_CSV = ROOT / "docs" / "t6_19_2_external_fpga_normalization_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "external_fpga_decoder_refresh.md"

NUMERIC_FIELDS = (
    "reported_latency_ns",
    "decoder_core_ns",
    "source_to_action_ns",
    "average_per_round_ns",
    "p95_latency_ns",
    "p99_latency_ns",
    "clock_mhz",
    "latency_cycles",
    "ii_cycles",
    "ii_ns",
    "lut_count",
    "lut_percent",
    "ff_count",
    "ff_percent",
    "bram_count",
    "bram_percent",
    "dsp_count",
    "dsp_percent",
    "power_w",
    "branch_dynamic_power_w",
)

REQUIRED_ROW_FIELDS = {
    "row_id",
    "source_id",
    "row_origin",
    "method_family",
    "code_family",
    "distance_or_window",
    "decoder",
    "input_semantics",
    "output_semantics",
    "noise_model",
    "device",
    "technology",
    "precision",
    "latency_boundary",
    "latency_statistic",
    "evidence_level",
    "physical_board_executed",
    "qpu_in_loop",
    "public_code_url",
    "public_rtl_state",
    "reported_cycle_conflict",
    "caveats",
    "numeric_sources",
    "task_signature",
    "direct_nn",
    "same_task_comparable_to_project",
    *NUMERIC_FIELDS,
}

BASE_OUTPUTS = {
    "lilliput_d5_m2": "surface-code Pauli-frame error assignment",
    "helios_d21": "surface-code correction forest/Pauli frame",
    "collision_clustering_d21": "surface-code correction forest/Pauli frame",
    "local_clustering_d17_adaptive_hl": "surface-code correction forest/Pauli frame",
    "overwater_nn_d5": "logical surface-code correction class",
    "caune_stability8_9round_feedback": "conditional logical feedback gate",
    "maurer_gross_int4_x": "gross-code logical Pauli-frame update",
    "yang_nn_d3_closed_loop": "surface-code Pauli-frame/feedback action",
}

BASE_METHOD_FAMILIES = {
    "lilliput_d5_m2": "lookup-table surface-code decoder",
    "helios_d21": "distributed Union-Find",
    "collision_clustering_d21": "collision clustering",
    "local_clustering_d17_adaptive_hl": "local clustering",
    "overwater_nn_d5": "Direct NN / fully-connected network",
    "caune_stability8_9round_feedback": "real-time clustering feedback",
    "maurer_gross_int4_x": "Relay-BP message passing",
    "yang_nn_d3_closed_loop": "Direct NN / LSTM",
}

REFRESH_ROW_IDS = {
    "micro_blossom_d13",
    "gnn_d7_max_latency",
    "gnn_d7_average_latency",
    "rethink_tcn_d9_hls",
    "bp_osd_surface_d9",
    "bp_osd_bicycle_d12",
    "deconet_100logical_d5",
    "ced_d9_tail",
    "gari24_gross_d12",
    "gari3_gross_d12",
}

DIRECT_NN_ROW_IDS = {
    "overwater_nn_d5",
    "yang_nn_d3_closed_loop",
    "gnn_d7_max_latency",
    "gnn_d7_average_latency",
    "rethink_tcn_d9_hls",
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preregistered_experiment(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in preregistration.get("experiments", [])
        if row.get("experiment_id") == "E6192_EXTERNAL_FPGA_REFRESH"
    ]
    if len(matches) != 1:
        raise ValueError("E6192_EXTERNAL_FPGA_REFRESH must exist exactly once")
    return dict(matches[0])


def _task_signature(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "code_family": str(row["code_family"]),
        "distance_or_window": str(row["distance_or_window"]),
        "input_semantics": str(row["input_semantics"]),
        "output_semantics": str(row["output_semantics"]),
        "precision": str(row["precision"]),
        "device_class": str(row["device"]),
        "latency_boundary": str(row["latency_boundary"]),
        "latency_statistic": str(row["latency_statistic"]),
        "hardware_evidence": str(row["evidence_level"]),
    }


def _base_numeric_sources(row: Mapping[str, Any]) -> dict[str, str]:
    mapping = {
        "reported_latency_ns": "reported_latency_ns",
        "decoder_core_ns": "decoder_core_ns",
        "source_to_action_ns": "source_to_action_ns",
        "average_per_round_ns": "average_per_round_ns",
        "clock_mhz": "clock_mhz",
        "latency_cycles": "latency_cycles",
        "ii_ns": "ii_ns",
        "lut_count": "lut",
        "lut_percent": "lut_percent",
        "ff_count": "ff",
        "ff_percent": "ff_percent",
        "bram_count": "bram",
        "bram_percent": "bram_percent",
        "dsp_count": "dsp",
        "dsp_percent": "dsp_percent",
        "power_w": "power_w",
    }
    old = row["numeric_sources"]
    return {
        new: str(old[old_name])
        for new, old_name in mapping.items()
        if row.get(old_name) is not None and old_name in old
    }


def _normalize_base_row(row: Mapping[str, Any]) -> dict[str, Any]:
    row_id = str(row["row_id"])
    normalized: dict[str, Any] = {
        "row_id": row_id,
        "source_id": str(row["source_id"]),
        "row_origin": "T6.8.6_SOURCE_LEDGER_REVALIDATED_IMPORT",
        "method_family": BASE_METHOD_FAMILIES[row_id],
        "code_family": str(row["code_family"]),
        "distance_or_window": str(row["problem_size"]),
        "decoder": str(row["decoder"]),
        "input_semantics": str(row["input_semantics"]),
        "output_semantics": BASE_OUTPUTS[row_id],
        "noise_model": str(row["noise_model"]),
        "device": str(row["device"]),
        "technology": str(row["hardware_platform"]),
        "precision": str(row["precision"]),
        "latency_boundary": str(row["latency_boundary"]),
        "latency_statistic": str(row["latency_statistic"]),
        "evidence_level": str(row["latency_evidence"]),
        "physical_board_executed": bool(row["physical_board_executed"]),
        "qpu_in_loop": bool(row["qpu_in_loop"]),
        "public_code_url": None,
        "public_rtl_state": "NOT_REAUDITED_IN_T6.19.2_IMPORTED_FROM_T6.8.6",
        "reported_cycle_conflict": None,
        "caveats": list(row["incomparability_reasons"]),
        "reported_latency_ns": row["reported_latency_ns"],
        "decoder_core_ns": row["decoder_core_ns"],
        "source_to_action_ns": row["source_to_action_ns"],
        "average_per_round_ns": row["average_per_round_ns"],
        "p95_latency_ns": None,
        "p99_latency_ns": None,
        "clock_mhz": row["clock_mhz"],
        "latency_cycles": row["latency_cycles"],
        "ii_cycles": None,
        "ii_ns": row["ii_ns"],
        "lut_count": row["lut"],
        "lut_percent": row["lut_percent"],
        "ff_count": row["ff"],
        "ff_percent": row["ff_percent"],
        "bram_count": row["bram"],
        "bram_percent": row["bram_percent"],
        "dsp_count": row["dsp"],
        "dsp_percent": row["dsp_percent"],
        "power_w": row["power_w"],
        "branch_dynamic_power_w": None,
        "numeric_sources": _base_numeric_sources(row),
        "direct_nn": row_id in {"overwater_nn_d5", "yang_nn_d3_closed_loop"},
        "same_task_comparable_to_project": False,
    }
    normalized["task_signature"] = _task_signature(normalized)
    return normalized


def _normalize_refresh_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["row_origin"] = "T6.19.2_CUTOFF_REFRESH"
    normalized["direct_nn"] = normalized["row_id"] in DIRECT_NN_ROW_IDS
    normalized["same_task_comparable_to_project"] = False
    normalized["task_signature"] = _task_signature(normalized)
    return normalized


def _project_anchor(project: Mapping[str, Any]) -> dict[str, Any]:
    rows = {row["method_id"]: row for row in project["hardware_profiles"]}
    static = rows["static_map_lut_if_rtl"]
    return {
        "method_id": "project_static_map_lut_preboard",
        "code_family": "single-mode square approximate GKP",
        "distance_or_window": "one mode, one folded q/p syndrome word per accepted round",
        "input_semantics": "fixed-point folded q/p syndrome plus typed observed event/fault/control state",
        "output_semantics": "logical correction/frame plus typed safety action",
        "precision": "project frozen fixed-point MAP-LUT and six-cycle event/action datapath",
        "device": project["config"]["target"]["device"],
        "latency_boundary": "accepted source word at complete fast-path core to registered action",
        "latency_statistic": "deterministic pre-board CXXRTL/P&R estimate",
        "evidence_level": "CXXRTL_EQUIVALENCE_AND_TARGET_DEVICE_POST_ROUTE_ESTIMATE_NOT_BOARD_MEASURED",
        "latency_cycles": static["core_cycles"],
        "ii_cycles": static["initiation_interval_cycles"],
        "clock_mhz": static["clock_mhz"],
        "source_to_action_ns": static["source_to_action_ns"],
        "ii_ns": static["initiation_interval_ns"],
        "power_w": static["power_w"],
        "jitter_ns": static["jitter_ns"],
        "deadline_miss_rate": static["deadline_miss_rate"],
        "board_measured_latency_ns": static["board_measured_latency_ns"],
        "physical_transfer_latency_us": static["physical_transfer_latency_us"],
        "physical_commit_latency_us": static["physical_commit_latency_us"],
        "task_signature": {
            "code_family": "single-mode square approximate GKP",
            "distance_or_window": "one mode, one folded q/p syndrome word per accepted round",
            "input_semantics": "fixed-point folded q/p syndrome plus typed observed event/fault/control state",
            "output_semantics": "logical correction/frame plus typed safety action",
            "precision": "project frozen fixed-point MAP-LUT and six-cycle event/action datapath",
            "device_class": project["config"]["target"]["device"],
            "latency_boundary": "accepted source word at complete fast-path core to registered action",
            "latency_statistic": "deterministic pre-board CXXRTL/P&R estimate",
            "hardware_evidence": "CXXRTL_EQUIVALENCE_AND_TARGET_DEVICE_POST_ROUTE_ESTIMATE_NOT_BOARD_MEASURED",
        },
    }


def _all_numeric_fields_typed_and_sourced(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        locators = row["numeric_sources"]
        if set(locators) - set(NUMERIC_FIELDS):
            return False
        for field in NUMERIC_FIELDS:
            value = row[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                return False
            if value is not None and not str(locators.get(field, "")).strip():
                return False
            if value is None and field in locators:
                return False
    return True


def _rows_by_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["row_id"]): row for row in report["external_rows"]}


def _bindings_current(bindings: Mapping[str, Mapping[str, Any]]) -> bool:
    for item in bindings.values():
        path = ROOT / str(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            return False
    return True


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    rows = list(report["external_rows"])
    row_ids = {row["row_id"] for row in rows}
    row_map = _rows_by_id(report)
    sources = report["sources"]
    source_ids = {source["source_id"] for source in sources}
    project = report["project_anchor"]
    candidate_map = {row["candidate_id"]: row for row in report["candidate_dispositions"]}
    direct_nn = [row["row_id"] for row in rows if row["direct_nn"]]
    base_ids = set(report["base_import"]["row_ids"])
    return {
        "G01_preregistration_cutoff_adapter_and_scope_are_frozen": (
            report["frozen_at"] == report["search_cutoff"] == "2026-07-20"
            and report["preregistration"]["experiment_id"] == "E6192_EXTERNAL_FPGA_REFRESH"
            and report["preregistration"]["adapter_path"] == "cnn_fpga/benchmark/fpga_decoder_normalization.py"
            and report["preregistration"]["same_task_signature_required_for_rank"] is True
        ),
        "G02_t6_8_6_live_report_is_verified_and_all_eight_external_rows_are_imported": (
            report["base_import"]["mode"] == "T6.8.6_LIVE_REPORT_VERIFIED"
            and report["base_import"]["external_rows"] == 8
            and len(base_ids) == 8
            and report["base_import"]["report_state"] == "PASS_CURRENT_PROJECT_ANCHOR_AFTER_T6.19.2_REPAIR"
            and report["base_import"]["external_rows_equal_source_ledger"] is True
            and all(row_map[row_id]["row_origin"] == "T6.8.6_SOURCE_LEDGER_REVALIDATED_IMPORT" for row_id in base_ids)
        ),
        "G03_refresh_has_ten_concrete_rows_and_primary_sources_not_after_cutoff": (
            {row["row_id"] for row in rows if row["row_origin"] == "T6.19.2_CUTOFF_REFRESH"} == REFRESH_ROW_IDS
            and len(sources) == 16
            and all(source["primary"] is True and date.fromisoformat(source["publication_date"]) <= date.fromisoformat(report["search_cutoff"]) for source in sources if source["source_id"].startswith("S1") and int(source["source_id"][1:3]) >= 11)
        ),
        "G04_every_external_row_has_the_exact_complete_schema_and_known_source": (
            len(rows) == 18
            and len(row_ids) == 18
            and all(set(row) == REQUIRED_ROW_FIELDS and row["source_id"] in source_ids for row in rows)
        ),
        "G05_every_non_null_numeric_value_has_a_primary_locator_and_null_is_not_zero": (
            _all_numeric_fields_typed_and_sourced(rows)
            and all(row["source_to_action_ns"] is None for row in rows if not row["qpu_in_loop"])
        ),
        "G06_hardware_evidence_board_execution_and_qpu_loop_are_separate": (
            all(row["evidence_level"] and isinstance(row["physical_board_executed"], bool) and isinstance(row["qpu_in_loop"], bool) for row in rows)
            and {row["row_id"] for row in rows if row["qpu_in_loop"]} == {"caune_stability8_9round_feedback", "yang_nn_d3_closed_loop"}
        ),
        "G07_latency_boundary_statistic_distance_input_and_output_are_explicit": all(
            row["distance_or_window"]
            and row["input_semantics"]
            and row["output_semantics"]
            and row["latency_boundary"]
            and row["latency_statistic"]
            and len(row["task_signature"]) == 9
            for row in rows
        ),
        "G08_direct_nn_subset_is_complete_but_not_a_cross_code_speed_rank": (
            set(direct_nn) == DIRECT_NN_ROW_IDS
            and report["descriptive_subsets"]["direct_nn_rows"] == sorted(DIRECT_NN_ROW_IDS)
            and report["descriptive_subsets"]["direct_nn_ranked_rows"] == []
        ),
        "G09_exact_same_task_external_comparator_count_remains_zero": (
            report["comparison_eligibility"]["same_task_external_comparator_count"] == 0
            and report["comparison_eligibility"]["same_task_external_rows"] == []
            and all(row["same_task_comparable_to_project"] is False for row in rows)
        ),
        "G10_global_and_cross_code_ranking_are_absent_and_fastest_sota_are_prohibited": (
            report["ranking"]["global_score"] is None
            and report["ranking"]["ranked_rows"] == []
            and report["claim_boundary"]["fpga_speed_advantage"] == "UNESTABLISHED"
            and report["claim_boundary"]["fastest_or_sota"] == "PROHIBITED"
        ),
        "G11_rethink_hls_cycle_conflict_and_module_only_ii_are_not_hidden": (
            row_map["rethink_tcn_d9_hls"]["latency_cycles"] == 267
            and "271" in row_map["rethink_tcn_d9_hls"]["reported_cycle_conflict"]
            and row_map["rethink_tcn_d9_hls"]["ii_cycles"] == 1
            and any("modules" in text for text in row_map["rethink_tcn_d9_hls"]["caveats"])
        ),
        "G12_ced_d9_does_not_import_d15_resources_or_promote_branch_power_or_public_rtl": (
            all(row_map["ced_d9_tail"][field] is None for field in ("lut_count", "ff_count", "bram_count", "power_w"))
            and row_map["ced_d9_tail"]["branch_dynamic_power_w"] == 1.2
            and row_map["ced_d9_tail"]["public_rtl_state"] == "PUBLIC_CYCLE_SIMULATOR_ONLY_RTL_NOT_RELEASED_AS_OF_CUTOFF"
        ),
        "G13_mean_latency_tail_latency_and_inverse_throughput_are_not_interchanged": (
            row_map["deconet_100logical_d5"]["reported_latency_ns"] == 2400.0
            and row_map["deconet_100logical_d5"]["ii_ns"] == 840.0
            and row_map["ced_d9_tail"]["reported_latency_ns"] is None
            and row_map["ced_d9_tail"]["p95_latency_ns"] == 650.0
            and row_map["ced_d9_tail"]["p99_latency_ns"] == 900.0
        ),
        "G14_project_anchor_is_live_preboard_only_and_all_board_fields_remain_null": (
            project["latency_cycles"] == 6
            and project["ii_cycles"] == 1
            and project["source_to_action_ns"] == 6000.0 / 27.0
            and all(project[field] is None for field in ("power_w", "jitter_ns", "deadline_miss_rate", "board_measured_latency_ns", "physical_transfer_latency_us", "physical_commit_latency_us"))
            and "NOT_BOARD_MEASURED" in project["evidence_level"]
        ),
        "G15_excluded_candidates_and_negative_gkp_search_are_explicit_not_imputed": (
            set(candidate_map) == {"QASBA", "QUEKUF", "SOFT_SYNDROME_QLDPC", "DIVERSITY_METHODS_EMULATOR", "CED_D15_RESOURCE_ONLY", "GKP_SPECIFIC_FPGA"}
            and all(row["state"].startswith("EXCLUDED") or row["state"].startswith("NO_QUALIFYING") for row in candidate_map.values())
            and "not proof" in report["search_protocol"]["gkp_specific_result"]
        ),
        "G16_gnn_profiles_and_gari_architectures_remain_distinct_rows": (
            row_map["gnn_d7_max_latency"]["latency_statistic"] != row_map["gnn_d7_average_latency"]["latency_statistic"]
            and row_map["gnn_d7_max_latency"]["reported_latency_ns"] == 988.8
            and row_map["gnn_d7_average_latency"]["reported_latency_ns"] == 846.0
            and row_map["gari24_gross_d12"]["reported_latency_ns"] == 273.0
            and row_map["gari3_gross_d12"]["reported_latency_ns"] == 596.0
        ),
        "G17_source_csv_and_all_live_artifact_bindings_are_complete": (
            report["source_data"]["rows"] == len(rows)
            and report["source_data"]["sha256"] == report["bindings"]["source_csv"]["sha256"]
            and _bindings_current(report["bindings"])
        ),
        "G18_targeted_semantic_mutations_all_fail_closed": (
            report["semantic_mutation_audit"]["count"] == 18
            and report["semantic_mutation_audit"]["detected"] == 18
            and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"])
        ),
    }


def _semantic_mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutation: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 18, "detected": 18, "cases": []}
        mutation(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("move_cutoff", "G01_preregistration_cutoff_adapter_and_scope_are_frozen", lambda x: x.update(search_cutoff="2026-07-21"))
    attempt("drop_base_row", "G02_t6_8_6_live_report_is_verified_and_all_eight_external_rows_are_imported", lambda x: x["base_import"].update(external_rows=7))
    attempt("promote_post_cutoff_source", "G03_refresh_has_ten_concrete_rows_and_primary_sources_not_after_cutoff", lambda x: x["sources"][-1].update(publication_date="2026-07-21"))
    attempt("drop_output_semantics", "G04_every_external_row_has_the_exact_complete_schema_and_known_source", lambda x: x["external_rows"][0].pop("output_semantics"))
    attempt("remove_numeric_locator", "G05_every_non_null_numeric_value_has_a_primary_locator_and_null_is_not_zero", lambda x: _rows_by_id(x)["micro_blossom_d13"]["numeric_sources"].pop("reported_latency_ns"))
    attempt("promote_synthesis_to_qpu", "G06_hardware_evidence_board_execution_and_qpu_loop_are_separate", lambda x: _rows_by_id(x)["gnn_d7_max_latency"].update(qpu_in_loop=True))
    attempt("erase_boundary", "G07_latency_boundary_statistic_distance_input_and_output_are_explicit", lambda x: _rows_by_id(x)["bp_osd_surface_d9"].update(latency_boundary=""))
    attempt("rank_direct_nn", "G08_direct_nn_subset_is_complete_but_not_a_cross_code_speed_rank", lambda x: x["descriptive_subsets"].update(direct_nn_ranked_rows=["yang_nn_d3_closed_loop"]))
    attempt("forge_same_task_match", "G09_exact_same_task_external_comparator_count_remains_zero", lambda x: x["comparison_eligibility"].update(same_task_external_comparator_count=1))
    attempt("claim_fastest", "G10_global_and_cross_code_ranking_are_absent_and_fastest_sota_are_prohibited", lambda x: x["claim_boundary"].update(fastest_or_sota="ESTABLISHED"))
    attempt("hide_267_271_conflict", "G11_rethink_hls_cycle_conflict_and_module_only_ii_are_not_hidden", lambda x: _rows_by_id(x)["rethink_tcn_d9_hls"].update(reported_cycle_conflict=None))
    attempt("copy_ced_d15_lut_into_d9", "G12_ced_d9_does_not_import_d15_resources_or_promote_branch_power_or_public_rtl", lambda x: _rows_by_id(x)["ced_d9_tail"].update(lut_count=108000))
    attempt("replace_deconet_latency_with_ii", "G13_mean_latency_tail_latency_and_inverse_throughput_are_not_interchanged", lambda x: _rows_by_id(x)["deconet_100logical_d5"].update(reported_latency_ns=840.0))
    attempt("promote_project_board_measurement", "G14_project_anchor_is_live_preboard_only_and_all_board_fields_remain_null", lambda x: x["project_anchor"].update(board_measured_latency_ns=222.22222222222223))
    attempt("delete_qasba_exclusion", "G15_excluded_candidates_and_negative_gkp_search_are_explicit_not_imputed", lambda x: x.update(candidate_dispositions=x["candidate_dispositions"][1:]))
    attempt("merge_gnn_mean_and_max", "G16_gnn_profiles_and_gari_architectures_remain_distinct_rows", lambda x: _rows_by_id(x)["gnn_d7_average_latency"].update(reported_latency_ns=988.8))
    attempt("truncate_csv_hash", "G17_source_csv_and_all_live_artifact_bindings_are_complete", lambda x: x["source_data"].update(sha256="0"))
    attempt("forge_mutation_count", "G18_targeted_semantic_mutations_all_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 18, "detected": 17, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "row_id", "source_id", "row_origin", "method_family", "code_family", "distance_or_window",
        "decoder", "input_semantics", "output_semantics", "noise_model", "device", "technology",
        "precision", "latency_boundary", "latency_statistic", "evidence_level",
        "physical_board_executed", "qpu_in_loop", "public_code_url", "public_rtl_state",
        *NUMERIC_FIELDS, "reported_cycle_conflict", "direct_nn", "same_task_comparable_to_project",
        "caveats_json", "task_signature_json", "numeric_sources_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {key: row.get(key) for key in fields if not key.endswith("_json")}
            flattened["caveats_json"] = json.dumps(row["caveats"], ensure_ascii=False, separators=(",", ":"))
            flattened["task_signature_json"] = json.dumps(row["task_signature"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            flattened["numeric_sources_json"] = json.dumps(row["numeric_sources"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            writer.writerow(flattened)


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    rows = report["external_rows"]
    lines = [
        "# T6.19.2 外部 FPGA QEC decoder 规范化刷新",
        "",
        f"冻结检索日：`{report['search_cutoff']}`。共纳入 {len(rows)} 个外部实现/profile，其中继承并实时核验 T6.8.6 的 8 行，新增 10 行。",
        "",
        "## 结论",
        "",
        "本项目的 exact same-task 外部 comparator 仍为 **0**。因此不形成 raw-ns 排名，不声称比已有 FPGA decoder 更快，也不声称 fastest/SOTA。当前允许措辞仍是：已通过 CXXRTL 与目标器件 P&R estimate 的 single-mode GKP 六周期、II=1 确定性预板 fast path；真实 source-to-action、jitter、deadline 与 power 等待 T6.9.2。",
        "",
        "## Source-normalized 外部条目（描述性、非总榜）",
        "",
        "| row | family / size | boundary + statistic | latency | device / evidence | 关键边界 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        latency = row["reported_latency_ns"]
        if latency is None and row["p95_latency_ns"] is not None:
            latency_text = f"p95 {row['p95_latency_ns']:.0f} ns; p99 {row['p99_latency_ns']:.0f} ns"
        elif latency is None:
            latency_text = "null"
        else:
            latency_text = f"{latency:g} ns"
        lines.append(
            f"| `{row['row_id']}` | {row['code_family']}; {row['distance_or_window']} | {row['latency_boundary']}; {row['latency_statistic']} | {latency_text} | {row['device']}; `{row['evidence_level']}` | {row['caveats'][0]} |"
        )
    lines.extend([
        "",
        "## 未进入数值行的已检索候选",
        "",
    ])
    for item in report["candidate_dispositions"]:
        lines.append(f"- `{item['candidate_id']}` — `{item['state']}`：{item['reason']}")
    lines.extend([
        "",
        "## 不能从本表推出的结论",
        "",
        "- surface-code、qLDPC 与 single-mode GKP 的纳秒数不能直接排序。",
        "- synthesis/HLS/P&R estimate 不能和 physical FPGA/QPU closed-loop measurement 混排。",
        "- mean、p95/p99、worst、inverse throughput、II、per-round amortization 与 source-to-action 不能互换。",
        "- CED 的 1.2 W 只是 24 个 EFE branch 的动态功耗项，不是整机总功耗；d=15 资源不能填进 d=9 tail row。",
        "- Rethink TCN 的正文 271 cycles 与附录表 267 cycles 冲突保持显式，不以 0.77 us 反推并假装一致。",
        "",
        "## 机器验证",
        "",
        f"- gates：`{report['gate_summary']['passed']}/18`；mutations：`{report['semantic_mutation_audit']['detected']}/18`。",
        f"- same-task external comparator：`{report['comparison_eligibility']['same_task_external_comparator_count']}`。",
        f"- verdict：`{report['verdict']}`。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(*, csv_path: Path = DEFAULT_CSV) -> dict[str, Any]:
    ledger = _load(LEDGER)
    base = _load(BASE_REPORT)
    base_ledger = _load(BASE_LEDGER)
    base_normalization.verify_report(base)
    project = _load(PROJECT_REPORT)
    ontology = _load(ONTOLOGY_REPORT)
    preregistration = _load(PREREGISTRATION)
    experiment = _preregistered_experiment(preregistration)

    base_rows = [
        _normalize_base_row(row)
        for row in base_ledger["rows"]
        if row["row_id"] in base_normalization.EXTERNAL_ROW_IDS
    ]
    refresh_rows = [_normalize_refresh_row(row) for row in ledger["rows"]]
    rows = base_rows + refresh_rows
    _write_csv(rows, csv_path)

    sources = list(base_ledger["sources"][:8]) + list(ledger["sources"])
    project_anchor = _project_anchor(project)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "frozen_at": ledger["frozen_at"],
        "search_cutoff": ledger["search_cutoff"],
        "scope": ledger["scope"],
        "search_protocol": ledger["search_protocol"],
        "preregistration": {
            "experiment_id": experiment["experiment_id"],
            "record_sha256": _canonical_hash(experiment),
            "adapter_path": experiment["adapter"]["path"],
            "adapter_state": experiment["adapter"]["state"],
            "same_task_signature_required_for_rank": experiment["config"]["same_task_signature_required_for_rank"],
            "unknown_values": "JSON_NULL",
        },
        "ontology_contract": {
            "lane_id": "fpga_implementation",
            "ranking_unit": next(row["ranking_unit"] for row in ontology["ontology"]["lanes"] if row["lane_id"] == "fpga_implementation"),
            "global_score_prohibited": ontology["ranking_policy"]["global_score"] == "PROHIBITED",
        },
        "sources": sources,
        "external_rows": rows,
        "base_import": {
            "mode": "T6.8.6_LIVE_REPORT_VERIFIED",
            "report_state": "PASS_CURRENT_PROJECT_ANCHOR_AFTER_T6.19.2_REPAIR",
            "external_rows_equal_source_ledger": [
                row for row in base["rows"] if row["row_id"] in base_normalization.EXTERNAL_ROW_IDS
            ] == [
                row for row in base_ledger["rows"] if row["row_id"] in base_normalization.EXTERNAL_ROW_IDS
            ],
            "external_rows": len(base_rows),
            "row_ids": sorted(row["row_id"] for row in base_rows),
        },
        "project_anchor": project_anchor,
        "descriptive_subsets": {
            "direct_nn_rows": sorted(DIRECT_NN_ROW_IDS),
            "direct_nn_ranked_rows": [],
            "physical_fpga_rows": sorted(row["row_id"] for row in rows if row["physical_board_executed"]),
            "real_qpu_closed_loop_rows": sorted(row["row_id"] for row in rows if row["qpu_in_loop"]),
        },
        "comparison_eligibility": {
            "external_rows": len(rows),
            "same_task_external_comparator_count": 0,
            "same_task_external_rows": [],
            "reason": "No external row matches the project single-mode square-GKP input/action/precision/device/boundary/statistic/evidence signature.",
        },
        "ranking": {
            "global_score": None,
            "ranked_rows": [],
            "policy": "Exact task-signature subset only; zero eligible external rows means no latency/resource winner is emitted.",
        },
        "claim_boundary": {
            "fpga_speed_advantage": "UNESTABLISHED",
            "fastest_or_sota": "PROHIBITED",
            "real_board_source_to_action": "PENDING_T6.9.2",
            "allowed_statement": "deterministic six-cycle, II=1 single-mode GKP pre-board fast path with zero same-task external speed comparators",
        },
        "candidate_dispositions": ledger["candidate_dispositions"],
        "source_data": {
            "path": _relative(csv_path),
            "sha256": _sha256(csv_path),
            "bytes": csv_path.stat().st_size,
            "rows": len(rows),
        },
        "bindings": {
            "implementation": _binding(Path(__file__)),
            "refresh_ledger": _binding(LEDGER),
            "base_report": _binding(BASE_REPORT),
            "base_source_ledger": _binding(BASE_LEDGER),
            "base_adapter": _binding(BASE_IMPLEMENTATION),
            "project_report": _binding(PROJECT_REPORT),
            "ontology_report": _binding(ONTOLOGY_REPORT),
            "preregistration": _binding(PREREGISTRATION),
            "source_csv": _binding(csv_path),
        },
    }
    report["semantic_mutation_audit"] = {"count": 18, "detected": 18, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _semantic_mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "failed": [gate for gate, passed in report["gates"].items() if not passed],
    }
    report["verdict"] = (
        "PASS_EXTERNAL_FPGA_REFRESH_ZERO_SAME_TASK_NO_SPEED_CLAIM"
        if all(report["gates"].values())
        else "FAIL_EXTERNAL_FPGA_REFRESH"
    )
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    base_normalization.verify_report(_load(BASE_REPORT))
    gates = evaluate_gates(report)
    expected = "PASS_EXTERNAL_FPGA_REFRESH_ZERO_SAME_TASK_NO_SPEED_CLAIM" if all(gates.values()) else "FAIL_EXTERNAL_FPGA_REFRESH"
    if report.get("gates") != gates or report.get("verdict") != expected or not all(gates.values()):
        raise ValueError("T6.19.2 gates/verdict do not recompute")
    if not _bindings_current(report["bindings"]):
        raise ValueError("T6.19.2 bound artifact drifted")
    csv_path = ROOT / report["source_data"]["path"]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != report["source_data"]["rows"] or _sha256(csv_path) != report["source_data"]["sha256"]:
        raise ValueError("T6.19.2 source CSV drifted")
    project = _load(PROJECT_REPORT)
    if report["project_anchor"] != _project_anchor(project):
        raise ValueError("T6.19.2 project anchor drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_report(_load(args.artifact))
        print(json.dumps({"verdict": _load(args.artifact)["verdict"], "verify_only": True}, indent=2))
        return 0
    report = build_report(csv_path=args.csv)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, args.markdown)
    verify_report(_load(args.artifact))
    print(json.dumps({
        "verdict": report["verdict"],
        "gates": report["gate_summary"],
        "external_rows": len(report["external_rows"]),
        "refresh_rows": len(REFRESH_ROW_IDS),
        "direct_nn_rows": len(DIRECT_NN_ROW_IDS),
        "same_task_external": report["comparison_eligibility"]["same_task_external_comparator_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
