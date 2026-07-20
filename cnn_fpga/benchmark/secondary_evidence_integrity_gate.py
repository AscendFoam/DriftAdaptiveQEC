"""T6.19.3 six-lane auxiliary atlas and fail-closed integrity gate.

The atlas is deliberately descriptive.  It recomputes statistics from frozen
Source Data, verifies every upstream artifact live, and forbids any cross-lane
winner, Phase-6B claim upgrade, or board-measurement promotion.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import aqec_secondary_wallclock_replay as aqec
from cnn_fpga.benchmark import comparison_metric_ontology as ontology_module
from cnn_fpga.benchmark import external_fpga_decoder_refresh as external_fpga
from cnn_fpga.benchmark import gqf_paper_exact_reproduction as gqf_exact
from cnn_fpga.benchmark import gqf_route_a_matched_comparison_gate as gqf_gate
from cnn_fpga.benchmark import learned_model_eligibility_replay as learned
from cnn_fpga.benchmark import multimode_posterior_weighted_cpd as multimode
from cnn_fpga.benchmark import official_structured_cpd_reproduction as official_cpd
from cnn_fpga.benchmark import phase6c_preboard_profiles as preboard
from cnn_fpga.benchmark import route_a_v5_final_evidence_gate as phase6b_gate
from cnn_fpga.benchmark import secondary_experiment_preregistration as prereg_module
from cnn_fpga.benchmark import secondary_method_source_audit as source_audit
from cnn_fpga.benchmark import single_mode_cpd_equivalence as single_cpd
from cnn_fpga.benchmark import static_gkp_same_model_lane as static_lane
from cnn_fpga.benchmark import surface_gkp_cnot_reproduction as surface_cnot


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.19.3"
SCHEMA_VERSION = "t6.19.3-secondary-evidence-integrity-v1"
VERDICT = "PASS_AUX_COMPARISON_INTEGRITY"

DEFAULT_REPORT = ROOT / "docs/t6_19_3_secondary_evidence_integrity.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t6_19_3_secondary_evidence_integrity_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/secondary_comparison_atlas.md"
DEFAULT_FIGURE_STEM = ROOT / "docs/figures/t6_19_3_secondary_comparison_atlas"
IMPLEMENTATION = Path(__file__).resolve()

ONTOLOGY = ROOT / "docs/t6_16_2_comparison_ontology.json"
PREREG = ROOT / "docs/t6_16_3_secondary_preregistration.json"
PREREG_CONFIG = ROOT / "configs/literature/t6_16_3_secondary_preregistration.json"
PHASE6B = ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json"
BOARD_BLOCKER = ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json"

REPORTS = {
    "source_audit": ROOT / "docs/t6_16_1_secondary_method_source_audit.json",
    "ontology": ONTOLOGY,
    "preregistration": PREREG,
    "phase6b": PHASE6B,
    "single_static": ROOT / "docs/t6_8_1_static_gkp_same_model_lane.json",
    "single_cpd": ROOT / "docs/t6_17_1_single_mode_cpd_equivalence.json",
    "surface_cnot": ROOT / "docs/t6_17_2_noh_cnot_ci_ml_reproduction.json",
    "learned": ROOT / "docs/t6_17_3_learned_model_eligibility_replay.json",
    "gqf_exact": ROOT / "docs/t6_8_4_gqf_paper_exact_reproduction.json",
    "gqf_gate": ROOT / "docs/t6_8_5_gqf_route_a_matched_comparison_gate.json",
    "aqec": ROOT / "docs/t6_18_1_aqec_common_wallclock_replay.json",
    "official_cpd": ROOT / "docs/t6_18_2_official_structured_cpd_reproduction.json",
    "multimode": ROOT / "docs/t6_18_3_multimode_posterior_weighted_cpd.json",
    "preboard": ROOT / "docs/t6_19_1_project_preboard_profiles.json",
    "external_fpga": ROOT / "docs/t6_19_2_external_fpga_normalization.json",
    "board_blocker": BOARD_BLOCKER,
}

RAW = {
    "single_static": ROOT / "docs/t6_8_1_static_gkp_same_model_lane_source_data.csv",
    "single_cpd": ROOT / "docs/t6_17_1_single_mode_cpd_equivalence_source_data.csv",
    "surface_cnot": ROOT / "docs/t6_17_2_noh_cnot_ci_ml_reproduction_source_data.csv",
    "learned": ROOT / "docs/t6_17_3_learned_model_eligibility_replay_source_data.csv",
    "gqf_exact": ROOT / "docs/t6_8_4_gqf_reproduction_source_data.csv",
    "gqf_gate": ROOT / "docs/t6_8_5_gqf_route_a_matched_comparison_gate_source_data.csv",
    "aqec": ROOT / "docs/t6_18_1_aqec_common_wallclock_source_data.csv",
    "official_cpd": ROOT / "docs/t6_18_2_official_structured_cpd_source_data.csv",
    "multimode": ROOT / "docs/t6_18_3_multimode_posterior_weighted_cpd_source_data.csv",
    "preboard": ROOT / "docs/t6_19_1_project_preboard_profiles_source_data.csv",
    "external_fpga": ROOT / "docs/t6_19_2_external_fpga_normalization_source_data.csv",
}

LANES = (
    "single_mode_decoder",
    "surface_gkp_gate_outer_code",
    "multimode_structured_lattice_cpd",
    "controller_rl_nmf",
    "aqec_wallclock",
    "fpga_implementation",
)
VALUE_STATES = {
    "MEASURED_VALUE", "ESTIMATE_VALUE", "REPRODUCED_VALUE", "LITERATURE_VALUE",
    "NULL_NOT_REPORTED", "N_A_NOT_APPLICABLE", "FAILED", "NEGATIVE",
}
NO_VALUE_STATES = {"NULL_NOT_REPORTED", "N_A_NOT_APPLICABLE", "FAILED", "NEGATIVE"}
EVIDENCE_GRADES = {
    "LITERATURE_ONLY", "OFFICIAL_CODE_REPRODUCTION", "PROJECT_NATIVE_MATCHED",
    "INELIGIBLE", "BLOCKED", "NEGATIVE",
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _close(left: Any, right: Any, *, atol: float = 1e-15) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=atol))


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _metric_maps(ontology: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    metrics = {row["metric_id"]: row for row in ontology["ontology"]["metrics"]}
    allowed = {lane: {key for key, row in metrics.items() if lane in row["allowed_lanes"]} for lane in LANES}
    return metrics, allowed


def _parent_payloads() -> dict[str, dict[str, Any]]:
    return {key: _load(path) for key, path in REPORTS.items()}


def _verify_parents(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    verifiers = {
        "source_audit": lambda: source_audit.verify_report(parents["source_audit"]),
        "ontology": lambda: ontology_module.verify_report(parents["ontology"]),
        "preregistration": lambda: prereg_module.verify_report(parents["preregistration"]),
        "phase6b": lambda: phase6b_gate.validate_report(PHASE6B),
        "single_static": lambda: static_lane.verify_report(parents["single_static"]),
        "single_cpd": lambda: single_cpd.verify_report(parents["single_cpd"]),
        "surface_cnot": lambda: surface_cnot.verify_report(parents["surface_cnot"]),
        "learned": lambda: learned.verify_report(parents["learned"]),
        "gqf_exact": lambda: gqf_exact.verify_report(parents["gqf_exact"]),
        "gqf_gate": lambda: gqf_gate.verify_report(parents["gqf_gate"]),
        "aqec": lambda: aqec.verify_report(parents["aqec"]),
        "official_cpd": lambda: official_cpd.verify_report(parents["official_cpd"]),
        "multimode": lambda: multimode.verify_report(REPORTS["multimode"]),
        "preboard": lambda: preboard.verify_report(REPORTS["preboard"]),
        "external_fpga": lambda: external_fpga.verify_report(parents["external_fpga"]),
    }
    for key, verifier in verifiers.items():
        try:
            verifier()
            checks[key] = True
        except Exception:
            checks[key] = False
    return checks


@lru_cache(maxsize=1)
def _live_recomputations() -> dict[str, Any]:
    parents = _parent_payloads()

    static_rows = _csv(RAW["single_static"])
    static_values = {
        row["key"]: float(row["value"])
        for row in static_rows if row["row_type"] == "method_summary"
    }

    cnot_rows = _csv(RAW["surface_cnot"])
    cnot_recomputed: dict[str, Any] = {}
    for db in (9.0, 12.0, 13.0):
        rows = [row for row in cnot_rows if row["record_type"] == "seed_cluster" and float(row["squeezing_db"]) == db]
        trials = sum(int(row["trials"]) for row in rows)
        failures = [tuple(int(value) for value in row["failures"].split("/")) for row in rows]
        ci_failures = sum(value[0] for value in failures)
        ml_failures = sum(value[1] for value in failures)
        cnot_recomputed[str(db)] = {
            "trials": trials,
            "ci_failures": ci_failures,
            "ml_failures": ml_failures,
            "ci_probability": ci_failures / trials,
            "ml_probability": ml_failures / trials,
            "ci_wilson_95": surface_cnot._wilson(ci_failures, trials),
            "ml_wilson_95": surface_cnot._wilson(ml_failures, trials),
        }

    official_live = official_cpd._live_recomputation()
    threshold_recomputed = {
        "official_cpd_threshold": official_live["official"]["cpd_threshold_mean"],
        "official_analog_threshold": official_live["official"]["analog_threshold_mean"],
        "independent_cpd_threshold": official_live["thresholds"]["cpd"]["central_crossings"]["mean_adjacent_crossing"],
        "independent_analog_threshold": official_live["thresholds"]["analog_mwpm"]["central_crossings"]["mean_adjacent_crossing"],
        "independent_total_paired_trials": official_live["integrity"]["total_paired_trials"],
    }

    multimode_rows = _csv(RAW["multimode"])
    multimode_summary: dict[str, Any] = {}
    methods = ("static_euclidean", "weighted_static", "observed_only_posterior_predictive_weighted", "oracle_metric_upper_bound")
    for method in methods:
        seeds = [row for row in multimode_rows if row["row_type"] == "seed_summary" and row["method"] == method]
        windows = [row for row in multimode_rows if row["row_type"] == "window" and row["method"] == method]
        multimode_summary[method] = {
            "errors": sum(int(row["numerator"]) for row in seeds),
            "cycles": sum(int(row["denominator"]) for row in seeds),
            "p_L": sum(int(row["numerator"]) for row in seeds) / sum(int(row["denominator"]) for row in seeds),
            "worst_window_ler": max(float(row["value"]) for row in windows),
            "seed_clusters": len(seeds),
        }

    aqec_rows = _csv(RAW["aqec"])
    aqec_summary: dict[str, Any] = {}
    for cell_index, cell in enumerate(parents["aqec"]["cells"]):
        cell_id = cell["cell_id"]
        metric_rows = [row for row in aqec_rows if row["record_type"] == "seed_metric" and row["cell_id"] == cell_id and row["metric"] == "lifetime_gain_ratio"]
        by_anchor: dict[str, list[float]] = {}
        for row in metric_rows:
            by_anchor.setdefault(row["anchor"], []).append(float(row["value"]))
        indices, seed = aqec._bootstrap_indices(cell_index, len(by_anchor["idle_memory"]))
        aqec_summary[cell_id] = {
            "measurement_feedback": aqec._summary(by_anchor["measurement_feedback"], indices, seed),
            "autonomous": aqec._summary(by_anchor["autonomous"], indices, seed),
        }

    hardware_rows = [row for row in _csv(RAW["preboard"]) if row["row_type"] == "hardware_seed"]
    fmax = [float(row["achieved_fmax_mhz"]) for row in hardware_rows]
    hardware = {
        "seed_count": len(hardware_rows),
        "minimum_fmax_mhz": min(fmax),
        "median_fmax_mhz": float(np.median(fmax)),
        "maximum_fmax_mhz": max(fmax),
        "lut4_counts": [int(row["lut4_count"]) for row in hardware_rows],
        "ff_counts": [int(row["ff_count"]) for row in hardware_rows],
        "bram_counts": [int(row["bram_count"]) for row in hardware_rows],
        "dsp_counts": [int(row["dsp_count"]) for row in hardware_rows],
        "source_to_action_ns_at_27mhz": 6.0 / 27.0 * 1000.0,
        "initiation_interval_ns_at_27mhz": 1.0 / 27.0 * 1000.0,
    }

    external_rows = _csv(RAW["external_fpga"])
    external = {
        "row_count": len(external_rows),
        "direct_nn_count": sum(row["direct_nn"].lower() == "true" for row in external_rows),
        "same_task_comparable_count": sum(row["same_task_comparable_to_project"].lower() == "true" for row in external_rows),
        "physical_board_count": sum(row["physical_board_executed"].lower() == "true" for row in external_rows),
    }
    return {
        "single_static": static_values,
        "surface_cnot": cnot_recomputed,
        "structured_cpd": threshold_recomputed,
        "multimode": multimode_summary,
        "aqec": aqec_summary,
        "hardware": hardware,
        "external_fpga": external,
    }


def _cell(
    *, cell_id: str, lane_id: str, method_id: str, label: str, metric_id: str,
    value: float | int | None, value_state: str, evidence_grade: str,
    outcome_state: str, source_key: str, signature_id: str,
    ranking_eligible: bool = False, ci95: Sequence[float] | None = None,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "lane_id": lane_id,
        "method_id": method_id,
        "label": label,
        "metric_id": metric_id,
        "value": value,
        "ci95": list(ci95) if ci95 is not None else None,
        "value_state": value_state,
        "evidence_grade": evidence_grade,
        "outcome_state": outcome_state,
        "ranking_eligible_within_signature": ranking_eligible,
        "task_signature_id": signature_id,
        "source_report": _binding(REPORTS[source_key]),
        "raw_source": _binding(RAW[source_key]) if source_key in RAW else _binding(REPORTS[source_key]),
        "config": _binding(PREREG_CONFIG),
        "detail": dict(detail or {}),
    }


def _cells(parents: Mapping[str, Mapping[str, Any]], live: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    static_labels = {
        "standard_binning": "Standard binning", "static_joint_map": "Static joint MAP",
        "proposed_route_a": "Route A", "hidden_state_oracle": "Hidden-state oracle",
    }
    for method in static_labels:
        cells.append(_cell(
            cell_id=f"single_{method}_pl", lane_id=LANES[0], method_id=method,
            label=static_labels[method], metric_id="p_L", value=live["single_static"][method],
            value_state="REPRODUCED_VALUE", evidence_grade="PROJECT_NATIVE_MATCHED",
            outcome_state="UPPER_BOUND_ONLY" if method == "hidden_state_oracle" else "VALID_DESCRIPTIVE_VALUE",
            source_key="single_static", signature_id="single_formal_same_trace_v1",
            ranking_eligible=method != "hidden_state_oracle",
        ))
    cells.append(_cell(
        cell_id="single_route_a_vs_static_negative", lane_id=LANES[0], method_id="proposed_route_a",
        label="Route A > static MAP claim", metric_id="ler_reduction_fraction", value=None,
        value_state="NEGATIVE", evidence_grade="NEGATIVE", outcome_state="NO_GO_STATIC_SUPERIORITY",
        source_key="single_static", signature_id="single_formal_same_trace_v1",
        detail={"reason": "Route A aggregate p_L is higher and worst-window LER is materially worse."},
    ))

    for point in parents["surface_cnot"]["points"]:
        db = str(float(point["squeezing_db"]))
        for method, key in (("ci", "ci"), ("ml", "ml")):
            cells.append(_cell(
                cell_id=f"surface_{method}_{db.replace('.', 'p')}db", lane_id=LANES[1], method_id=method,
                label=f"{method.upper()} @ {db} dB", metric_id="cnot_failure_probability",
                value=live["surface_cnot"][db][f"{key}_probability"],
                ci95=live["surface_cnot"][db][f"{key}_wilson_95"],
                value_state="REPRODUCED_VALUE", evidence_grade="PROJECT_NATIVE_MATCHED",
                outcome_state="MATCHED_PAIRED_MONTE_CARLO", source_key="surface_cnot",
                signature_id="noh_surface_gkp_cnot_paired_v1", ranking_eligible=True,
                detail={"trials": live["surface_cnot"][db]["trials"]},
            ))
    cells.append(_cell(
        cell_id="surface_outer_code_threshold_na", lane_id=LANES[1], method_id="outer_code_threshold",
        label="Outer-code threshold", metric_id="squeezing_threshold_db", value=None,
        value_state="N_A_NOT_APPLICABLE", evidence_grade="INELIGIBLE",
        outcome_state="NOT_ESTIMATED_BY_GATE_LEVEL_EXPERIMENT", source_key="surface_cnot",
        signature_id="noh_surface_gkp_cnot_paired_v1",
    ))

    for method, label in (("cpd", "Official CPD"), ("analog_mwpm", "Official analog MWPM")):
        cells.append(_cell(
            cell_id=f"structured_official_{method}_threshold", lane_id=LANES[2], method_id=method,
            label=label, metric_id="noise_threshold_sigma",
            value=live["structured_cpd"]["official_cpd_threshold" if method == "cpd" else "official_analog_threshold"],
            value_state="REPRODUCED_VALUE", evidence_grade="OFFICIAL_CODE_REPRODUCTION",
            outcome_state="OFFICIAL_AGGREGATE_REANALYSIS", source_key="official_cpd",
            signature_id="surface_lattice_official_threshold_v1", ranking_eligible=True,
            detail={"declared_samples_per_point": 10_000_000},
        ))
    multimode_labels = {
        "static_euclidean": "Static Euclidean", "weighted_static": "Weighted static",
        "observed_only_posterior_predictive_weighted": "Observed-only predictive",
        "oracle_metric_upper_bound": "Metric oracle",
    }
    for method, label in multimode_labels.items():
        value = live["multimode"][method]["p_L"]
        cells.append(_cell(
            cell_id=f"multimode_{method}_pl", lane_id=LANES[2], method_id=method,
            label=label, metric_id="p_L", value=value, value_state="REPRODUCED_VALUE",
            evidence_grade="PROJECT_NATIVE_MATCHED", source_key="multimode",
            signature_id="project_multimode_drift_v1",
            outcome_state="UPPER_BOUND_ONLY" if "oracle" in method else "VALID_MATCHED_VALUE",
            ranking_eligible="oracle" not in method,
            detail={"worst_window_ler": live["multimode"][method]["worst_window_ler"], "seed_clusters": live["multimode"][method]["seed_clusters"]},
        ))
    cells.append(_cell(
        cell_id="multimode_weighted_static_negative", lane_id=LANES[2], method_id="weighted_static",
        label="Weighted-static advantage", metric_id="p_L", value=None, value_state="NEGATIVE",
        evidence_grade="NEGATIVE", outcome_state="WORSE_THAN_STATIC_EUCLIDEAN",
        source_key="multimode", signature_id="project_multimode_drift_v1",
    ))

    cells.extend([
        _cell(cell_id="controller_learned_same_task", lane_id=LANES[3], method_id="learned_candidates",
              label="Eligible learned controller", metric_id="ler_reduction_fraction", value=None,
              value_state="NULL_NOT_REPORTED", evidence_grade="INELIGIBLE", outcome_state="ZERO_OF_16_SAME_TASK_ELIGIBLE",
              source_key="learned", signature_id="controller_source_audit_v1",
              detail={"candidate_families": parents["learned"]["eligibility_summary"]["candidate_families"], "eligible": 0}),
        _cell(cell_id="controller_gqf_exact", lane_id=LANES[3], method_id="puviani_gqf_nmf",
              label="Exact GQF/NMF reproduction", metric_id="lifetime_gain_ratio", value=None,
              value_state="FAILED", evidence_grade="BLOCKED", outcome_state="NO_GO_SOURCE_INCOMPLETE",
              source_key="gqf_exact", signature_id="puviani_gqf_exact_v1"),
        _cell(cell_id="controller_gqf_route_a_rank", lane_id=LANES[3], method_id="gqf_vs_route_a",
              label="GQF vs Route A rank", metric_id="ler_reduction_fraction", value=None,
              value_state="NEGATIVE", evidence_grade="NEGATIVE", outcome_state="INELIGIBLE_TASK_SIGNATURE",
              source_key="gqf_gate", signature_id="cross_task_forbidden_v1"),
    ])

    for cell_id, values in live["aqec"].items():
        for anchor, label in (("measurement_feedback", "Feedback"), ("autonomous", "Autonomous")):
            summary = values[anchor]
            cells.append(_cell(
                cell_id=f"aqec_{cell_id}_{anchor}", lane_id=LANES[4], method_id=anchor,
                label=f"{cell_id}: {label}", metric_id="lifetime_gain_ratio", value=summary["mean"],
                ci95=summary["ci95"], value_state="REPRODUCED_VALUE", evidence_grade="PROJECT_NATIVE_MATCHED",
                outcome_state="ACTIVE_QEC_BELOW_IDLE", source_key="aqec", signature_id="aqec_common_wallclock_v1",
                ranking_eligible=True, detail={"n_seed_clusters": summary["n_seed_clusters"]},
            ))
    cells.append(_cell(
        cell_id="aqec_official_protocol_blocked", lane_id=LANES[4], method_id="official_aqec_protocol",
        label="Official full AQEC protocol", metric_id="lifetime_gain_ratio", value=None,
        value_state="FAILED", evidence_grade="BLOCKED", outcome_state="OFFICIAL_PROTOCOL_REPRODUCTION_BLOCKED",
        source_key="aqec", signature_id="aqec_official_protocol_v1",
    ))

    hardware = live["hardware"]
    for metric, value, label in (
        ("latency_ns", hardware["source_to_action_ns_at_27mhz"], "6-cycle source-to-action"),
        ("initiation_interval_ns", hardware["initiation_interval_ns_at_27mhz"], "II=1"),
        ("lut_count", int(np.median(hardware["lut4_counts"])), "Median post-route LUT4"),
        ("ff_count", int(np.median(hardware["ff_counts"])), "Median post-route FF"),
        ("bram_count", int(np.median(hardware["bram_counts"])), "Median post-route BRAM"),
        ("dsp_count", int(np.median(hardware["dsp_counts"])), "Median post-route DSP"),
    ):
        cells.append(_cell(
            cell_id=f"fpga_project_{metric}", lane_id=LANES[5], method_id="static_map_lut_if_rtl",
            label=label, metric_id=metric, value=value, value_state="ESTIMATE_VALUE",
            evidence_grade="PROJECT_NATIVE_MATCHED", outcome_state="POST_ROUTE_ESTIMATE_NOT_BOARD_MEASURED",
            source_key="preboard", signature_id="project_static_map_fast_path_v1", ranking_eligible=False,
        ))

    external_metric_map = {
        "reported_latency_ns": "latency_ns",
        "clock_mhz": "clock_mhz",
        "latency_cycles": "latency_cycles",
        "ii_ns": "initiation_interval_ns",
        "lut_count": "lut_count",
        "ff_count": "ff_count",
        "bram_count": "bram_count",
        "dsp_count": "dsp_count",
        "power_w": "power_w",
    }
    for external in parents["external_fpga"]["external_rows"]:
        for source_field, metric_id in external_metric_map.items():
            value = external[source_field]
            cells.append(_cell(
                cell_id=f"fpga_external_{external['row_id']}_{metric_id}",
                lane_id=LANES[5], method_id=external["row_id"], label=external["decoder"],
                metric_id=metric_id, value=value,
                value_state="LITERATURE_VALUE" if value is not None else "NULL_NOT_REPORTED",
                evidence_grade="LITERATURE_ONLY",
                outcome_state="SOURCE_REPORTED_DESCRIPTIVE_NOT_RANKED" if value is not None else "FIELD_NOT_REPORTED",
                source_key="external_fpga", signature_id=f"external_{external['row_id']}", ranking_eligible=False,
                detail={
                    "source_field": source_field, "source_id": external["source_id"],
                    "code_family": external["code_family"], "distance_or_window": external["distance_or_window"],
                    "device": external["device"], "latency_boundary": external["latency_boundary"],
                    "latency_statistic": external["latency_statistic"], "hardware_evidence": external["evidence_level"],
                    "physical_board_executed": external["physical_board_executed"], "direct_nn": external["direct_nn"],
                    "same_task_comparable_to_project": external["same_task_comparable_to_project"],
                },
            ))
    cells.extend([
        _cell(cell_id="fpga_external_same_task", lane_id=LANES[5], method_id="external_fpga_decoders",
              label="Same-task external comparator", metric_id="latency_ns", value=None,
              value_state="NEGATIVE", evidence_grade="NEGATIVE", outcome_state="ZERO_SAME_TASK_COMPARATORS",
              source_key="external_fpga", signature_id="external_fpga_normalized_v1",
              detail=live["external_fpga"]),
        _cell(cell_id="fpga_direct_nn_same_task", lane_id=LANES[5], method_id="direct_nn",
              label="Same-task Direct NN RTL", metric_id="latency_ns", value=None,
              value_state="N_A_NOT_APPLICABLE", evidence_grade="INELIGIBLE", outcome_state="NO_ELIGIBLE_DIRECT_NN_RTL",
              source_key="external_fpga", signature_id="external_fpga_normalized_v1",
              detail={"descriptive_direct_nn_rows": live["external_fpga"]["direct_nn_count"]}),
        _cell(cell_id="fpga_board_latency_null", lane_id=LANES[5], method_id="static_map_lut_if_rtl",
              label="Board-measured latency", metric_id="latency_ns", value=None,
              value_state="NULL_NOT_REPORTED", evidence_grade="BLOCKED", outcome_state="WAITING_FOR_PHYSICAL_BOARD",
              source_key="preboard", signature_id="project_static_map_fast_path_v1"),
    ])
    return cells


def _write_source_data(cells: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "lane_id", "cell_id", "method_id", "label", "metric_id", "value", "ci_low", "ci_high",
        "value_state", "evidence_grade", "outcome_state", "ranking_eligible_within_signature",
        "task_signature_id", "source_report_path", "source_report_sha256", "raw_source_path",
        "raw_source_sha256", "config_path", "config_sha256", "detail_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            ci = cell["ci95"] or (None, None)
            writer.writerow({
                "lane_id": cell["lane_id"], "cell_id": cell["cell_id"], "method_id": cell["method_id"],
                "label": cell["label"], "metric_id": cell["metric_id"], "value": cell["value"],
                "ci_low": ci[0], "ci_high": ci[1], "value_state": cell["value_state"],
                "evidence_grade": cell["evidence_grade"], "outcome_state": cell["outcome_state"],
                "ranking_eligible_within_signature": cell["ranking_eligible_within_signature"],
                "task_signature_id": cell["task_signature_id"],
                "source_report_path": cell["source_report"]["path"], "source_report_sha256": cell["source_report"]["sha256"],
                "raw_source_path": cell["raw_source"]["path"], "raw_source_sha256": cell["raw_source"]["sha256"],
                "config_path": cell["config"]["path"], "config_sha256": cell["config"]["sha256"],
                "detail_json": json.dumps(cell["detail"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            })


def _render_figure(cells: Sequence[Mapping[str, Any]], stem: Path) -> dict[str, dict[str, Any]]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 6.5, "axes.titlesize": 7.2, "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.6,
    })
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 7.35), constrained_layout=True)
    palette = {"blue": "#3B6FB6", "orange": "#E18A2B", "teal": "#2A9D8F", "grey": "#787878", "red": "#B64E4E"}

    def panel(ax: Any, letter: str, title: str) -> None:
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")

    ax = axes[0, 0]
    panel(ax, "a", "Single-mode decoder (same formal trace)")
    selected = [c for c in cells if c["cell_id"] in {"single_standard_binning_pl", "single_static_joint_map_pl", "single_proposed_route_a_pl", "single_hidden_state_oracle_pl"}]
    ax.bar(range(4), [c["value"] * 1e3 for c in selected], color=[palette["grey"], palette["blue"], palette["orange"], palette["teal"]], edgecolor="black", linewidth=0.35)
    ax.set_xticks(range(4), ["Binning", "Static\nMAP", "Route A", "Oracle"])
    ax.set_ylabel(r"$p_L$ ($\times 10^{-3}$)")
    ax.text(0.02, 0.96, "Route A superiority: NO-GO", transform=ax.transAxes, va="top", color=palette["red"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0})

    ax = axes[0, 1]
    panel(ax, "b", "Surface–GKP CNOT reproduction")
    for method, color, marker in (("ci", palette["grey"], "o"), ("ml", palette["blue"], "s")):
        rows = sorted([c for c in cells if c["lane_id"] == LANES[1] and c["method_id"] == method and c["value"] is not None], key=lambda c: float(c["label"].split("@")[1].split()[0]))
        x = [float(c["label"].split("@")[1].split()[0]) for c in rows]
        y = [c["value"] for c in rows]
        lo = [c["value"] - c["ci95"][0] for c in rows]
        hi = [c["ci95"][1] - c["value"] for c in rows]
        ax.errorbar(x, y, yerr=[lo, hi], marker=marker, color=color, label=method.upper(), capsize=2, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Squeezing (dB)")
    ax.set_ylabel("CNOT failure probability")
    ax.legend(frameon=False)

    ax = axes[0, 2]
    panel(ax, "c", "Multimode drift lane (same project task)")
    order = ["static_euclidean", "weighted_static", "observed_only_posterior_predictive_weighted", "oracle_metric_upper_bound"]
    rows = [next(c for c in cells if c["cell_id"] == f"multimode_{method}_pl") for method in order]
    ax.bar(range(4), [c["value"] for c in rows], color=[palette["grey"], palette["orange"], palette["blue"], palette["teal"]], edgecolor="black", linewidth=0.35)
    ax.set_xticks(range(4), ["Static", "Weighted\nstatic", "Observed-only\npredictive", "Oracle"])
    ax.set_ylabel(r"Aggregate $p_L$")
    ax.text(0.02, 0.96, "32 seed clusters; predictive wins 32/32", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0})

    ax = axes[1, 0]
    panel(ax, "d", "Controller / learned-module eligibility")
    ax.set_xlim(0, 1); ax.set_ylim(0, 3); ax.axis("off")
    status_rows = [
        ("Learned candidates", "0/16 same-task eligible", palette["grey"]),
        ("Puviani GQF/NMF", "exact replay blocked", palette["red"]),
        ("GQF vs Route A", "cross-task rank forbidden", palette["orange"]),
    ]
    for index, (left, right, color) in enumerate(status_rows):
        y = 2.55 - index
        ax.add_patch(plt.Rectangle((0.02, y - 0.34), 0.96, 0.58, facecolor=color, alpha=0.14, edgecolor=color, linewidth=0.8))
        ax.text(0.06, y, left, fontweight="bold", va="center", fontsize=6.0)
        ax.text(0.96, y, right.replace(" same-task ", " same-task\n").replace(" cross-task ", " cross-task\n"), ha="right", va="center", fontsize=5.5)

    ax = axes[1, 1]
    panel(ax, "e", "AQEC common-wall-clock replay")
    ids = list(_live_recomputations()["aqec"])
    x = np.arange(len(ids), dtype=float)
    for offset, anchor, label, color, marker in ((-0.12, "measurement_feedback", "Feedback", palette["blue"], "o"), (0.12, "autonomous", "Autonomous", palette["orange"], "s")):
        rows = [next(c for c in cells if c["cell_id"] == f"aqec_{cell_id}_{anchor}") for cell_id in ids]
        y = np.asarray([c["value"] for c in rows])
        lo = y - np.asarray([c["ci95"][0] for c in rows]); hi = np.asarray([c["ci95"][1] for c in rows]) - y
        ax.errorbar(x + offset, y, yerr=[lo, hi], linestyle="none", marker=marker, color=color, capsize=2, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, label="Idle break-even")
    ax.set_xticks(x, [value.replace("cutoff", "c").replace("medium", "med") for value in ids], rotation=35, ha="right")
    ax.set_ylabel("Lifetime / idle lifetime")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=5.2, columnspacing=0.8, handletextpad=0.4)

    ax = axes[1, 2]
    panel(ax, "f", "FPGA evidence boundary")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.77, "6 cycles / II=1", "222.2 ns / 37.0 ns @ 27 MHz", palette["blue"]),
        (0.48, "3-seed post-route estimate", "LUT4 3377–3387; Fmax ≥41.02 MHz", palette["teal"]),
        (0.19, "18 external implementations", "0 same-task; board fields remain null", palette["red"]),
    ]
    for y, title, subtitle, color in boxes:
        ax.add_patch(plt.Rectangle((0.03, y - 0.11), 0.94, 0.2, facecolor=color, alpha=0.13, edgecolor=color, linewidth=0.8))
        ax.text(0.07, y + 0.025, title, fontweight="bold", va="center")
        ax.text(0.07, y - 0.045, subtitle, va="center")
    fig.suptitle("Six-lane auxiliary comparison atlas — no cross-lane ranking", fontsize=8.3, fontweight="bold")

    paths = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("svg", "pdf", "tiff", "png")}
    fig.savefig(paths["svg"], bbox_inches="tight")
    # Matplotlib emits trailing spaces in multiline SVG path data.  Normalize
    # them so repository whitespace checks remain useful and deterministic.
    svg_text = paths["svg"].read_text(encoding="utf-8")
    paths["svg"].write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {suffix: _binding(path) for suffix, path in paths.items()}


def _preregistered_experiment(parent: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [row for row in parent["experiments"] if row["experiment_id"] == "E6193_AUXILIARY_ATLAS_INTEGRITY"]
    if len(matches) != 1:
        raise ValueError("E6193_AUXILIARY_ATLAS_INTEGRITY must exist exactly once")
    return matches[0]


def _bindings_current(bindings: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        (ROOT / row["path"]).is_file()
        and _sha256(ROOT / row["path"]) == row["sha256"]
        and (ROOT / row["path"]).stat().st_size == row["bytes"]
        for row in bindings.values()
    )


def _recomputations_match(report: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, bool]:
    stored = report["recomputations"]
    return {
        "single": stored["single_static"] == live["single_static"],
        "cnot": stored["surface_cnot"] == live["surface_cnot"],
        "threshold": stored["structured_cpd"] == live["structured_cpd"],
        "multimode": stored["multimode"] == live["multimode"],
        "aqec": stored["aqec"] == live["aqec"],
        "hardware": stored["hardware"] == live["hardware"],
        "external": stored["external_fpga"] == live["external_fpga"],
    }


def evaluate_gates(report: Mapping[str, Any], *, verify_parents: bool = True) -> dict[str, bool]:
    parents = _parent_payloads()
    ontology = parents["ontology"]
    metrics, allowed = _metric_maps(ontology)
    cells = report["cells"]
    lane_set = {cell["lane_id"] for cell in cells}
    prereg = _preregistered_experiment(parents["preregistration"])
    recomputed = _recomputations_match(report, _live_recomputations())
    parent_checks = _verify_parents(parents) if verify_parents else report["parent_verification"]
    board = parents["board_blocker"]
    source_path = ROOT / report["source_data"]["path"]
    figure_paths = [ROOT / row["path"] for row in report["figures"].values()]
    gates = {
        "G01_exact_six_lanes_present": lane_set == set(LANES) and all(any(cell["lane_id"] == lane for cell in cells) for lane in LANES),
        "G02_preregistered_facets_and_readonly_adapter": tuple(prereg["config"]["facets"]) == LANES and prereg["execution_type"] == "READONLY_INTEGRITY_GATE" and report["preregistration_snapshot"] == {"experiment_id": prereg["experiment_id"], "execution_type": prereg["execution_type"], "facets": prereg["config"]["facets"], "stopping_rule": prereg["stopping_rule"]},
        "G03_value_states_and_evidence_grades_closed": all(cell["value_state"] in VALUE_STATES and cell["evidence_grade"] in EVIDENCE_GRADES for cell in cells),
        "G04_value_presence_matches_state": all((cell["value"] is None) == (cell["value_state"] in NO_VALUE_STATES) and (cell["ci95"] is None or cell["value"] is not None) for cell in cells),
        "G05_metric_is_legal_for_lane": all(cell["metric_id"] in metrics and cell["metric_id"] in allowed[cell["lane_id"]] for cell in cells),
        "G06_every_cell_has_current_report_raw_config_hash": all(_bindings_current({"report": cell["source_report"], "raw": cell["raw_source"], "config": cell["config"]}) for cell in cells) and _bindings_current(report["parent_bindings"]) and _bindings_current(report["raw_bindings"]) and _bindings_current({"implementation": report["implementation_binding"]}),
        "G07_cell_ids_unique": len(cells) == len({cell["cell_id"] for cell in cells}),
        "G08_null_na_failed_negative_are_explicit": NO_VALUE_STATES <= {cell["value_state"] for cell in cells},
        "G09_no_global_score_winner_or_cross_lane_rank": report["ranking_policy"] == {"global_score": False, "global_winner": None, "cross_lane_ranking": False},
        "G10_oracles_are_nonranking": all(not cell["ranking_eligible_within_signature"] for cell in cells if "oracle" in cell["method_id"]),
        "G11_literature_never_promoted_to_reproduction": all(not (cell["evidence_grade"] == "LITERATURE_ONLY" and cell["value_state"] not in {"LITERATURE_VALUE", "NULL_NOT_REPORTED"}) for cell in cells),
        "G12_estimate_never_promoted_to_measured": all(not (cell["value_state"] == "MEASURED_VALUE" and cell["outcome_state"].endswith("ESTIMATE_NOT_BOARD_MEASURED")) for cell in cells),
        "G13_phase6b_verdict_unchanged": report["phase6b_snapshot"]["verdict"] == "NO_GO_V5_EARLY_HEADROOM_STOP" and parents["phase6b"]["verdict"] == report["phase6b_snapshot"]["verdict"],
        "G14_t6_9_2_remains_blocked_with_null_measurements": report["board_blocker_snapshot"]["verdict"] == "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL" and board["verdict"] == report["board_blocker_snapshot"]["verdict"] and all(value is None for value in board["measured_results"].values()),
        "G15_all_parent_verifiers_pass_live": all(parent_checks.values()) and report["parent_verification"] == parent_checks,
        "G16_single_mode_raw_recomputation_exact": recomputed["single"],
        "G17_cnot_counts_probabilities_and_wilson_ci_recompute": recomputed["cnot"],
        "G18_structured_cpd_thresholds_recompute": recomputed["threshold"],
        "G19_multimode_counts_and_worst_window_recompute": recomputed["multimode"],
        "G20_aqec_cluster_bootstrap_ci_recomputes": recomputed["aqec"],
        "G21_fpga_timing_and_resource_profiles_recompute": recomputed["hardware"],
        "G22_external_fpga_counts_recompute_and_no_same_task_rank": recomputed["external"] and report["recomputations"]["external_fpga"]["same_task_comparable_count"] == 0,
        "G23_source_data_is_one_row_per_cell_and_hash_bound": source_path.is_file() and _sha256(source_path) == report["source_data"]["sha256"] and len(_csv(source_path)) == len(cells) == report["source_data"]["rows"],
        "G24_figure_bundle_hash_bound_and_svg_text_editable": len(figure_paths) == 4 and _bindings_current(report["figures"]) and "<text" in (ROOT / report["figures"]["svg"]["path"]).read_text(encoding="utf-8"),
    }
    return gates


def _mutation_audit(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in evaluate_gates(report, verify_parents=False):
        mutated = copy.deepcopy(report)
        if gate == "G01_exact_six_lanes_present": mutated["cells"] = [c for c in mutated["cells"] if c["lane_id"] != LANES[0]]
        elif gate == "G02_preregistered_facets_and_readonly_adapter": mutated["preregistration_snapshot"]["execution_type"] = "MUTATING"
        elif gate == "G03_value_states_and_evidence_grades_closed": mutated["cells"][0]["value_state"] = "UNKNOWN"
        elif gate == "G04_value_presence_matches_state": next(c for c in mutated["cells"] if c["value_state"] in NO_VALUE_STATES)["value"] = 1.0
        elif gate == "G05_metric_is_legal_for_lane": mutated["cells"][0]["metric_id"] = "noise_threshold_sigma"
        elif gate == "G06_every_cell_has_current_report_raw_config_hash": mutated["cells"][0]["raw_source"]["sha256"] = "0" * 64
        elif gate == "G07_cell_ids_unique": mutated["cells"][1]["cell_id"] = mutated["cells"][0]["cell_id"]
        elif gate == "G08_null_na_failed_negative_are_explicit": mutated["cells"] = [c for c in mutated["cells"] if c["value_state"] != "NEGATIVE"]
        elif gate == "G09_no_global_score_winner_or_cross_lane_rank": mutated["ranking_policy"]["global_score"] = True
        elif gate == "G10_oracles_are_nonranking": next(c for c in mutated["cells"] if "oracle" in c["method_id"])["ranking_eligible_within_signature"] = True
        elif gate == "G11_literature_never_promoted_to_reproduction":
            candidate = mutated["cells"][0]; candidate["evidence_grade"] = "LITERATURE_ONLY"; candidate["value_state"] = "REPRODUCED_VALUE"
        elif gate == "G12_estimate_never_promoted_to_measured": next(c for c in mutated["cells"] if c["value_state"] == "ESTIMATE_VALUE")["value_state"] = "MEASURED_VALUE"
        elif gate == "G13_phase6b_verdict_unchanged": mutated["phase6b_snapshot"]["verdict"] = "GO"
        elif gate == "G14_t6_9_2_remains_blocked_with_null_measurements": mutated["board_blocker_snapshot"]["verdict"] = "UNBLOCKED"
        elif gate == "G15_all_parent_verifiers_pass_live": mutated["parent_verification"]["phase6b"] = False
        elif gate == "G16_single_mode_raw_recomputation_exact": mutated["recomputations"]["single_static"]["static_joint_map"] += 1e-4
        elif gate == "G17_cnot_counts_probabilities_and_wilson_ci_recompute": mutated["recomputations"]["surface_cnot"]["9.0"]["ci_failures"] += 1
        elif gate == "G18_structured_cpd_thresholds_recompute": mutated["recomputations"]["structured_cpd"]["official_cpd_threshold"] += 0.01
        elif gate == "G19_multimode_counts_and_worst_window_recompute": mutated["recomputations"]["multimode"]["static_euclidean"]["errors"] += 1
        elif gate == "G20_aqec_cluster_bootstrap_ci_recomputes": mutated["recomputations"]["aqec"][next(iter(mutated["recomputations"]["aqec"]))]["autonomous"]["ci95"][0] += 0.01
        elif gate == "G21_fpga_timing_and_resource_profiles_recompute": mutated["recomputations"]["hardware"]["source_to_action_ns_at_27mhz"] = 6.0
        elif gate == "G22_external_fpga_counts_recompute_and_no_same_task_rank": mutated["recomputations"]["external_fpga"]["same_task_comparable_count"] = 1
        elif gate == "G23_source_data_is_one_row_per_cell_and_hash_bound": mutated["source_data"]["sha256"] = "0" * 64
        elif gate == "G24_figure_bundle_hash_bound_and_svg_text_editable": mutated["figures"]["svg"]["sha256"] = "0" * 64
        failed = [name for name, passed in evaluate_gates(mutated, verify_parents=False).items() if not passed]
        rows.append({"mutation_id": f"M_{gate[1:3]}", "target_gate": gate, "detected": gate in failed, "failed_gates": failed})
    return rows


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    counts = {lane: sum(cell["lane_id"] == lane for cell in report["cells"]) for lane in LANES}
    lines = [
        "# T6.19.3 六条 lane 非主排名对比图谱", "",
        f"- 完整性结论：`{report['verdict']}`", "- 定位：只读、辅助、within-lane；没有全局分数或跨 lane 冠军。",
        "- Phase 6B：保持 `NO_GO_V5_EARLY_HEADROOM_STOP`；T6.9.2 仍等待真板。", "",
        "## 图谱合同", "",
        "每个单元都绑定 source report、原始 Source Data 与冻结配置的 SHA-256；数值、CI、threshold、timing 和计数均从原始数据重算。",
        "", "| lane | cells | 结论边界 |", "|---|---:|---|",
    ]
    conclusions = {
        LANES[0]: "Route A 未胜 static MAP；oracle 仅上界",
        LANES[1]: "同任务 Noh CNOT 复现中 ML 低于 CI",
        LANES[2]: "official threshold 与 project drift 为不同 signature；均只在各自 signature 内解释",
        LANES[3]: "16 个 learned family 无同任务可排名项；GQF exact blocked",
        LANES[4]: "六个 common-wall-clock cell 的 active-QEC lifetime ratio 均低于 1",
        LANES[5]: "六周期/II=1 与资源均为仿真/布局布线估计；18 个外部实现中同任务为 0",
    }
    lines.extend(f"| `{lane}` | {counts[lane]} | {conclusions[lane]} |" for lane in LANES)
    lines.extend([
        "", "## 统计与图形 QA", "",
        "- Noh CNOT：paired Monte Carlo；每点 Wilson 95% CI；trial 数见 Source Data。",
        "- Multimode：32 个 seed cluster；aggregate p_L 由 error/cycle 原始计数重算。",
        "- AQEC：每 cell 24 个 seed cluster；20,000 次 cluster bootstrap 95% CI。",
        "- FPGA：3 个 P&R seed；27 MHz 下 6-cycle latency 与 II=1 换算；非板测。",
        "- Python/matplotlib 单后端；SVG 保留 editable text，同时导出 PDF/TIFF/PNG。", "",
        "## 产物", "",
        f"- `{report['source_data']['path']}`", f"- `{report['figures']['svg']['path']}`",
        f"- `{report['figures']['pdf']['path']}`", f"- `{report['figures']['tiff']['path']}`",
        f"- `{report['figures']['png']['path']}`", "",
        "不得将该 atlas 用作跨 code、跨任务、跨 latency boundary 或 estimate-vs-measured 的速度/性能排名。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    report_path: Path = DEFAULT_REPORT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
    markdown_path: Path = DEFAULT_MARKDOWN,
    figure_stem: Path = DEFAULT_FIGURE_STEM,
) -> dict[str, Any]:
    parents = _parent_payloads()
    parent_verification = _verify_parents(parents)
    if not all(parent_verification.values()):
        raise ValueError(f"upstream verification failed: {[key for key, value in parent_verification.items() if not value]}")
    live = _live_recomputations()
    cells = _cells(parents, live)
    _write_source_data(cells, source_data_path)
    figures = _render_figure(cells, figure_stem)
    experiment = _preregistered_experiment(parents["preregistration"])
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "six-lane read-only auxiliary atlas; no global score and no Phase-6B/hardware claim promotion",
        "preregistration_snapshot": {"experiment_id": experiment["experiment_id"], "execution_type": experiment["execution_type"], "facets": experiment["config"]["facets"], "stopping_rule": experiment["stopping_rule"]},
        "ranking_policy": {"global_score": False, "global_winner": None, "cross_lane_ranking": False},
        "phase6b_snapshot": {"task_id": parents["phase6b"]["task_id"], "verdict": parents["phase6b"]["verdict"], "analysis_sha256": parents["phase6b"]["analysis_sha256"]},
        "board_blocker_snapshot": {"task_id": parents["board_blocker"]["task_id"], "verdict": parents["board_blocker"]["verdict"]},
        "parent_verification": parent_verification,
        "implementation_binding": _binding(IMPLEMENTATION),
        "parent_bindings": {key: _binding(path) for key, path in REPORTS.items()},
        "raw_bindings": {key: _binding(path) for key, path in RAW.items()},
        "cells": cells,
        "recomputations": live,
        "source_data": {**_binding(source_data_path), "rows": len(cells)},
        "figures": figures,
        "claim_boundary": {
            "allowed": "within-lane, exact-signature descriptive or matched comparisons with explicit evidence states",
            "forbidden": ["global winner", "cross-lane score", "literature as reproduction", "estimate as measured", "upgrade T6.15.5", "unblock T6.9.2"],
        },
    }
    gates = evaluate_gates(report)
    failed = [key for key, value in gates.items() if not value]
    report["gates"] = gates
    report["semantic_mutation_audit"] = _mutation_audit(report)
    report["gate_summary"] = {"passed": len(gates) - len(failed), "failed": failed}
    report["verdict"] = VERDICT if not failed and all(row["detected"] for row in report["semantic_mutation_audit"]) else "FAIL_AUX_COMPARISON_INTEGRITY"
    report["analysis_sha256"] = _canonical_sha256({key: report[key] for key in ("ranking_policy", "phase6b_snapshot", "board_blocker_snapshot", "implementation_binding", "parent_bindings", "raw_bindings", "cells", "recomputations", "source_data", "figures", "claim_boundary", "gates", "semantic_mutation_audit", "verdict")})
    _atomic_json(report, report_path)
    _write_markdown(report, markdown_path)
    return report


def verify_report(report: Mapping[str, Any] | None = None, report_path: Path = DEFAULT_REPORT) -> dict[str, bool]:
    value = dict(report) if report is not None else _load(report_path)
    gates = evaluate_gates(value)
    expected_hash = _canonical_sha256({key: value[key] for key in ("ranking_policy", "phase6b_snapshot", "board_blocker_snapshot", "implementation_binding", "parent_bindings", "raw_bindings", "cells", "recomputations", "source_data", "figures", "claim_boundary", "gates", "semantic_mutation_audit", "verdict")})
    checks = {
        "identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION,
        "gates": gates == value.get("gates") and all(gates.values()),
        "mutations": len(value.get("semantic_mutation_audit", [])) == len(gates) and all(row["detected"] for row in value["semantic_mutation_audit"]),
        "verdict": value.get("verdict") == VERDICT,
        "analysis_hash": value.get("analysis_sha256") == expected_hash,
    }
    if not all(checks.values()):
        raise ValueError(f"T6.19.3 verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    args = parser.parse_args(argv)
    report = build_report(args.report, args.source_data, args.markdown, args.figure_stem)
    verify_report(report)
    print(json.dumps({"verdict": report["verdict"], "cells": len(report["cells"]), "gates": report["gate_summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
