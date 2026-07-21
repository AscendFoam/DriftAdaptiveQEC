"""Build the T6.26.3 non-transferable dual-lane paper evidence contract."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark import converged_hardware_lane_qualification as hardware_gate
from cnn_fpga.benchmark import converged_long_rtl_qualification as long_gate
from cnn_fpga.benchmark import converged_rtl_formal as formal_gate
from cnn_fpga.benchmark import dual_evidence_lane_contract as dual_gate
from cnn_fpga.benchmark import multimode_causal_headroom as headroom_gate
from cnn_fpga.benchmark import multimode_posterior_weighted_cpd as opened_gate


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.26.3"
SCHEMA_VERSION = "t6.26.3-dual-lane-evidence-matrix-v1"
VERDICT = "PASS_NONTRANSFERABLE_DUAL_LANE_EVIDENCE_AND_FIGURE_CONTRACT"
RUNNER = Path(__file__).resolve()
CONFIG = ROOT / "configs/phase6d/t6_26_3_dual_lane_evidence_matrix.json"
BOARD = ROOT / "docs/new_task_board.md"
REPORT = ROOT / "docs/t6_26_3_dual_lane_evidence_matrix.json"
SOURCE_DATA = ROOT / "docs/t6_26_3_dual_lane_evidence_source_data.csv"
MARKDOWN = ROOT / "docs/phase6d_dual_lane_evidence_matrix.md"

DUAL_REPORT = ROOT / "docs/t6_20_2_dual_evidence_lane_contract.json"
OPENED_REPORT = ROOT / "docs/t6_18_3_multimode_posterior_weighted_cpd.json"
HEADROOM_REPORT = ROOT / "docs/t6_20_4_multimode_causal_headroom.json"
FORMAL_REPORT = ROOT / "docs/t6_25_2_converged_rtl_formal.json"
LONG_REPORT = ROOT / "docs/t6_25_3_converged_long_rtl.json"
HARDWARE_REPORT = ROOT / "docs/t6_25_4_converged_hardware.json"

ARTIFACT_PATHS = {
    "implementation": RUNNER,
    "self_config": CONFIG,
    "cancellation_ledger": ROOT / "docs/phase6d_multimode_v1_cancellation_ledger.md",
    "dual_contract_report": DUAL_REPORT,
    "dual_contract_raw": ROOT / "docs/t6_20_2_dual_evidence_lane_contract_source_data.csv",
    "dual_contract_code": ROOT / "cnn_fpga/benchmark/dual_evidence_lane_contract.py",
    "mm_opened_report": OPENED_REPORT,
    "mm_opened_raw": ROOT / "docs/t6_18_3_multimode_posterior_weighted_cpd_source_data.csv",
    "mm_opened_config": ROOT / "configs/literature/t6_18_3_multimode_drift.json",
    "mm_opened_code": ROOT / "cnn_fpga/benchmark/multimode_posterior_weighted_cpd.py",
    "mm_official_source": ROOT / "third_party/LatticeAlgorithms.jl/src/gkp.jl",
    "mm_headroom_report": HEADROOM_REPORT,
    "mm_headroom_raw": ROOT / "runs/t6_20_4_causal_headroom_raw.json",
    "mm_headroom_source_data": ROOT / "docs/t6_20_4_multimode_causal_headroom_source_data.csv",
    "mm_headroom_config": ROOT / "configs/phase6d/t6_20_4_causal_headroom.json",
    "mm_headroom_code": ROOT / "cnn_fpga/benchmark/multimode_causal_headroom.py",
    "mm_headroom_source": ROOT / "scripts/run_t6_20_4_causal_headroom.jl",
    "rtl_formal_report": FORMAL_REPORT,
    "rtl_formal_raw": ROOT / "docs/t6_25_2_converged_rtl_formal_source_data.csv",
    "rtl_formal_config": ROOT / "configs/phase6d/t6_25_2_converged_rtl_formal.json",
    "rtl_formal_code": ROOT / "cnn_fpga/benchmark/converged_rtl_formal.py",
    "rtl_production_source": ROOT / "cnn_fpga/rtl/gkp_route_a_converged_production_top.sv",
    "rtl_long_report": LONG_REPORT,
    "rtl_long_raw": ROOT / "docs/t6_25_3_converged_long_rtl_source_data.csv",
    "rtl_long_config": ROOT / "configs/phase6d/t6_25_3_converged_long_rtl.json",
    "rtl_long_code": ROOT / "cnn_fpga/benchmark/converged_long_rtl_qualification.py",
    "rtl_long_source": ROOT / "cnn_fpga/runtime/converged_production_reference.py",
    "rtl_hardware_report": HARDWARE_REPORT,
    "rtl_hardware_raw": ROOT / "docs/t6_25_4_converged_hardware_source_data.csv",
    "rtl_hardware_config": ROOT / "configs/phase6d/t6_25_4_converged_hardware.json",
    "rtl_hardware_code": ROOT / "cnn_fpga/benchmark/converged_hardware_lane_qualification.py",
    "rtl_hardware_source": ROOT / "cnn_fpga/rtl/gkp_route_a_converged_synth_top.sv",
    "rtl_hardware_netlist": ROOT / "docs/t6_25_4_converged_synth_netlist.json",
    "board_blocker_report": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "board_blocker_code": ROOT / "cnn_fpga/benchmark/route_a_board_measurement_gate.py",
}

EVIDENCE_CATEGORIES = ("reports", "raw_data", "configs", "code", "sources")
CLAIM_STATES = {
    "RESULTS_ONLY_NONRANKING",
    "MANDATORY_NEGATIVE",
    "BLOCKED_NOT_RUN",
    "CURRENT_RESTRICTED",
    "PROHIBITED_POSITIVE",
    "DROPPED_ABSENT",
    "META_BOUNDARY",
}


class IntegrityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"not an object: {path}")
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
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return (
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and _sha256(path) == binding["sha256"]
    )


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _task_statuses(text: str) -> dict[str, str]:
    rows = re.findall(r"^\| (T[^| ]+) \| ([^|]+) \|", text, flags=re.MULTILINE)
    result: dict[str, str] = {}
    for task, status in rows:
        result.setdefault(task.strip(), status.strip())
    return result


def _board_snapshot() -> dict[str, Any]:
    config = _load(CONFIG)
    statuses = _task_statuses(BOARD.read_text(encoding="utf-8"))
    selected: dict[str, str] = {}
    for task, expected in config["required_task_statuses"].items():
        actual = statuses.get(task, "MISSING")
        selected[task] = "ACTIVE_OR_DONE" if expected == "ACTIVE_OR_DONE" and actual in {"In Progress", "Done"} else actual
    return {
        "path": _relative(BOARD),
        "statuses": selected,
        "canonical_sha256": _canonical_sha256(selected),
    }


def _evidence(
    *, reports: Sequence[str], raw_data: Sequence[str], configs: Sequence[str],
    code: Sequence[str], sources: Sequence[str], selectors: Sequence[str],
) -> dict[str, Any]:
    return {
        "reports": list(reports),
        "raw_data": list(raw_data),
        "configs": list(configs),
        "code": list(code),
        "sources": list(sources),
        "selectors": list(selectors),
    }


def _claim(
    claim_id: str, lane_id: str, state: str, metric_namespaces: Sequence[str],
    safe_wording: str, boundary: str, current_result: Any,
    evidence: Mapping[str, Any], forbidden_wording: Sequence[str],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "lane_id": lane_id,
        "state": state,
        "metric_namespaces": list(metric_namespaces),
        "safe_wording": safe_wording,
        "boundary": boundary,
        "current_result": current_result,
        "evidence": dict(evidence),
        "forbidden_wording": list(forbidden_wording),
        "wording_status": "SAFE",
    }


def _claims(
    opened: Mapping[str, Any], headroom: Mapping[str, Any], formal: Mapping[str, Any],
    long: Mapping[str, Any], hardware: Mapping[str, Any],
) -> list[dict[str, Any]]:
    opened_aggregate = opened["summaries"]["aggregate"]
    opened_comparison = opened["comparisons"]["aggregate"]["adaptive_vs_static_euclidean"]
    boot = headroom["paired_bootstrap"]
    common_mm_opened = _evidence(
        reports=["mm_opened_report"], raw_data=["mm_opened_raw"],
        configs=["self_config", "mm_opened_config"], code=["mm_opened_code"],
        sources=["mm_official_source"],
        selectors=["summaries.aggregate", "comparisons.aggregate", "formal_counts", "claim_boundary"],
    )
    common_mm_headroom = _evidence(
        reports=["mm_headroom_report"], raw_data=["mm_headroom_raw", "mm_headroom_source_data"],
        configs=["self_config", "mm_headroom_config"], code=["mm_headroom_code"],
        sources=["mm_headroom_source", "cancellation_ledger"],
        selectors=["strongest_development_baseline_selection", "paired_bootstrap", "headroom_gate", "scope"],
    )
    common_rtl = _evidence(
        reports=["rtl_formal_report", "rtl_long_report", "rtl_hardware_report"],
        raw_data=["rtl_formal_raw", "rtl_long_raw", "rtl_hardware_raw"],
        configs=["self_config", "rtl_formal_config", "rtl_long_config", "rtl_hardware_config"],
        code=["rtl_formal_code", "rtl_long_code", "rtl_hardware_code"],
        sources=["rtl_production_source", "rtl_long_source", "rtl_hardware_source"],
        selectors=["formal_results", "aggregate_python", "clock_model", "fmax_mhz", "resource_summary", "critical_paths"],
    )
    return [
        _claim(
            "MM_OPENED_TASK_LOCAL_GAIN", "MULTIMODE_SOFTWARE_ALGORITHM",
            "RESULTS_ONLY_NONRANKING", ["LER", "TAIL", "COMPUTE"],
            "On the opened project-native d=3 heteroscedastic drift extension, observed-only posterior-weighted CPD lowers paired LER and registered tail metrics relative to the two implemented static CPD baselines.",
            "This is task-local opened development evidence; it is not the Phase-6D strongest-baseline or frozen-benchmark SOTA result.",
            {
                "cycles": opened["formal_counts"]["total_physical_cycles"],
                "candidate_p_L": opened_aggregate["observed_only_posterior_predictive_weighted"]["p_L"],
                "static_euclidean_p_L": opened_aggregate["static_euclidean"]["p_L"],
                "absolute_improvement": opened_comparison["improvement"],
                "candidate_worst_window_ler": opened_aggregate["observed_only_posterior_predictive_weighted"]["worst_window_ler"],
                "candidate_cvar95_window_ler": opened_aggregate["observed_only_posterior_predictive_weighted"]["cvar95_window_ler"],
                "candidate_runtime_seconds": opened_aggregate["observed_only_posterior_predictive_weighted"]["runtime_seconds"],
                "candidate_seconds_per_decode": opened_aggregate["observed_only_posterior_predictive_weighted"]["seconds_per_decode"],
                "candidate_allocated_bytes_first_decode_max": opened_aggregate["observed_only_posterior_predictive_weighted"]["allocated_bytes_first_decode_max"],
            },
            common_mm_opened,
            ["multimode SOTA", "strongest deployable baseline", "implemented by the six-cycle RTL"],
        ),
        _claim(
            "MM_V1_CAUSAL_HEADROOM_NO_GO", "MULTIMODE_SOFTWARE_ALGORITHM",
            "MANDATORY_NEGATIVE", ["LER", "EVIDENCE_STATE"],
            "The preregistered Phase-6D v1 causal-headroom gate failed: the risk-aware action equals the strongest retained static-mixture exact-MLD baseline, giving zero paired relative improvement and zero lower confidence bound.",
            "All 13 development families and both retained baseline candidates remain visible; pilot and formal splits were not accessed.",
            {
                "strongest_baseline": headroom["strongest_development_baseline_selection"]["selected"],
                "baseline_p_L": boot["baseline_p_L"],
                "proposed_p_L": boot["proposed_p_L"],
                "relative_improvement_point": boot["relative_improvement_point"],
                "relative_improvement_lcb": boot["relative_improvement_lcb"],
                "formal_or_pilot_accessed": headroom["scope"]["formal_or_pilot_accessed"],
                "verdict": headroom["verdict"],
            },
            common_mm_headroom,
            ["passed the 10% gate", "delete the strongest baseline", "rescued by RTL safety"],
        ),
        _claim(
            "MM_FROZEN_BENCHMARK_SOTA_BLOCKED", "MULTIMODE_SOFTWARE_ALGORITHM",
            "BLOCKED_NOT_RUN", ["EVIDENCE_STATE"],
            "No frozen-benchmark multimode SOTA result exists because the preregistered v1 headroom gate failed and T6.24.5 was dropped without pilot or formal access.",
            "T6.18.3 remains a task-local positive comparator result and cannot replace the unrun strongest-baseline promotion gate.",
            None, common_mm_headroom,
            ["frozen-benchmark SOTA", "universal GKP SOTA", "formal confirmation"],
        ),
        _claim(
            "RTL_DETERMINISTIC_SIX_CYCLE_II1", "SINGLE_MODE_DETERMINISTIC_RTL",
            "CURRENT_RESTRICTED", ["LATENCY_CYCLES", "INITIATION_INTERVAL", "CXXRTL"],
            "The exact single-mode converged RTL has a six-cycle, II=1 pre-board source-to-action architecture and passed a one-million-cycle full-public-vector CXXRTL qualification with zero mismatch.",
            "Cycles are RTL/CXXRTL evidence; nanoseconds are a post-route clock model without transport, CDC, pins or physical jitter.",
            {
                "latency_cycles": hardware["clock_model"]["cycles"],
                "initiation_interval_cycles": hardware["clock_model"]["initiation_interval_cycles"],
                "cycles_qualified": long["aggregate_python"]["cycles"],
                "mismatches": sum(int(row["mismatches"]) for row in long["cxxrtl_families"]),
                "ii1_input_pairs": long["aggregate_python"]["ii1_input_pairs"],
                "ii1_output_pairs": long["aggregate_python"]["ii1_output_pairs"],
            },
            common_rtl,
            ["multimode decoder latency", "board-measured latency", "fastest FPGA decoder"],
        ),
        _claim(
            "RTL_ATOMIC_FAIL_CLOSED", "SINGLE_MODE_DETERMINISTIC_RTL",
            "CURRENT_RESTRICTED", ["PROPERTY", "CXXRTL"],
            "The exact converged production top passes the stated atomic versioned-bank, fail-closed, cover and mutation-closed pre-board property contract.",
            "The proof scope is the stated two-state RTL model and does not establish physical CDC, metastability or unbounded liveness.",
            {
                "formal_gates": formal["gate_summary"],
                "formal_mutations": formal["mutation_summary"],
                "long_gates": long["gate_summary"],
                "long_semantic_mutations": long["semantic_mutations"],
            },
            common_rtl,
            ["physical fail-safe proof", "multimode safety proof", "board verified"],
        ),
        _claim(
            "RTL_POST_ROUTE_ESTIMATE", "SINGLE_MODE_DETERMINISTIC_RTL",
            "CURRENT_RESTRICTED", ["POST_ROUTE", "RESOURCE", "LATENCY_CYCLES"],
            "The exact qualified top passes three-seed open-source GW2AR place-and-route at 27 MHz with the reported whole-harness Fmax and resource ranges.",
            "All three critical paths terminate in the observability fold, so Fmax is a conservative qualification-harness estimate rather than bare-core or board source-to-action speed.",
            {
                "fmax_mhz": hardware["fmax_mhz"],
                "resource_summary": hardware["resource_summary"],
                "critical_paths": hardware["critical_paths"],
                "clock_model": hardware["clock_model"],
            },
            common_rtl,
            ["bare-core Fmax", "vendor signoff", "measured power", "fastest"],
        ),
        _claim(
            "RTL_BOARD_MEASUREMENT_BLOCKED", "SINGLE_MODE_DETERMINISTIC_RTL",
            "BLOCKED_NOT_RUN", ["BOARD_NULL", "EVIDENCE_STATE"],
            "Physical-board correctness, latency, jitter, deadline, transport and power remain unavailable and null.",
            "No pre-board clock model, post-route estimate or analytic power sensitivity may fill a measured field.",
            hardware["measured_fields"],
            _evidence(
                reports=["rtl_hardware_report", "board_blocker_report"], raw_data=["rtl_hardware_raw"],
                configs=["self_config", "rtl_hardware_config"], code=["rtl_hardware_code", "board_blocker_code"],
                sources=["rtl_production_source", "rtl_hardware_source"],
                selectors=["measured_fields", "evidence_boundary"],
            ),
            ["board measured", "zero physical deadline miss", "measured power"],
        ),
        _claim(
            "RTL_SPEED_ADVANTAGE_PROHIBITED", "SINGLE_MODE_DETERMINISTIC_RTL",
            "PROHIBITED_POSITIVE", ["EVIDENCE_STATE"],
            "No faster or fastest claim is supported because no same-task physical-board comparator is available and the current Fmax belongs to an observability harness.",
            "The hardware contribution is deterministic, atomic and fail-closed pre-board architecture, not a cross-paper speed rank.",
            {"fastest_or_sota": hardware["evidence_boundary"]["fastest_or_sota"]},
            common_rtl,
            ["faster than existing FPGA decoders", "fastest", "SOTA latency"],
        ),
        _claim(
            "LEARNING_APPROXIMATION_DROPPED", "LEARNED_APPROXIMATION_EXTENSION",
            "DROPPED_ABSENT", ["STATUS", "EVIDENCE_STATE"],
            "CNN/student is absent from the primary Phase-6D result because the v1 headroom gate did not authorize a teacher, distillation, quantization or matched formal-retention run.",
            "Legacy CNN evidence remains an ablation only and cannot rescue either primary lane.",
            {"T6.26.1": "Dropped", "T6.26.2": "Dropped", "present_in_primary_rtl": False},
            _evidence(
                reports=["dual_contract_report", "mm_headroom_report", "rtl_hardware_report"],
                raw_data=["dual_contract_raw", "mm_headroom_source_data", "rtl_hardware_raw"],
                configs=["self_config", "mm_headroom_config", "rtl_hardware_config"],
                code=["dual_contract_code", "mm_headroom_code", "rtl_hardware_code"],
                sources=["cancellation_ledger", "rtl_hardware_source"],
                selectors=["lanes.LEARNED_APPROXIMATION_EXTENSION", "scope", "learning_extension"],
            ),
            ["CNN-centric primary contribution", "student proves SOTA", "student proves RTL safety"],
        ),
        _claim(
            "DUAL_LANE_NONTRANSFERABILITY", "META_CONTRACT",
            "META_BOUNDARY", ["EVIDENCE_STATE"],
            "Multimode software LER evidence and single-mode RTL deterministic/safety evidence are parallel but non-transferable; no weighted global score or cross-lane gate substitution is permitted.",
            "The learning extension is dependent and absent, not a third primary evidence lane.",
            {"global_weighted_score": None, "one_lane_cannot_satisfy_another_lane_gate": True},
            _evidence(
                reports=["dual_contract_report", "mm_headroom_report", "rtl_hardware_report"],
                raw_data=["dual_contract_raw", "mm_headroom_source_data", "rtl_hardware_raw"],
                configs=["self_config", "mm_headroom_config", "rtl_hardware_config"],
                code=["dual_contract_code", "mm_headroom_code", "rtl_hardware_code"],
                sources=["cancellation_ledger", "rtl_production_source"],
                selectors=["forbidden_transfers", "headroom_gate", "evidence_boundary"],
            ),
            ["combined LER-latency score", "RTL rescues multimode", "multimode validates RTL"],
        ),
    ]


def _element(
    element_id: str, panel_id: str, lane_id: str, metric_namespace: str,
    title: str, value: Any, evidence: Mapping[str, Any], annotation: str,
    forbidden_interpretation: Sequence[str],
) -> dict[str, Any]:
    return {
        "element_id": element_id,
        "panel_id": panel_id,
        "lane_id": lane_id,
        "metric_namespace": metric_namespace,
        "title": title,
        "value": value,
        "evidence": dict(evidence),
        "allowed_wording": annotation,
        "annotation": annotation,
        "forbidden_interpretation": list(forbidden_interpretation),
    }


def _figure_contract(claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {row["claim_id"]: row for row in claims}
    config = _load(CONFIG)

    def claim_evidence(claim_id: str) -> Mapping[str, Any]:
        return by_id[claim_id]["evidence"]

    opened = by_id["MM_OPENED_TASK_LOCAL_GAIN"]["current_result"]
    postroute = by_id["RTL_POST_ROUTE_ESTIMATE"]["current_result"]
    elements = [
        _element("MM-E1", "MULTIMODE_SOFTWARE", "MULTIMODE_SOFTWARE_ALGORITHM", "LER", "Opened task-local d=3 LER", {key: opened[key] for key in ("cycles", "candidate_p_L", "static_euclidean_p_L", "absolute_improvement")}, claim_evidence("MM_OPENED_TASK_LOCAL_GAIN"), "Opened development context; not strongest-baseline SOTA.", ["global SOTA", "hardware latency"]),
        _element("MM-E2", "MULTIMODE_SOFTWARE", "MULTIMODE_SOFTWARE_ALGORITHM", "TAIL", "Opened task-local registered tail", {key: opened[key] for key in ("candidate_worst_window_ler", "candidate_cvar95_window_ler")}, claim_evidence("MM_OPENED_TASK_LOCAL_GAIN"), "Non-overlapping 512-cycle windows under the frozen task-local rule.", ["general tail SOTA", "strongest-baseline tail"]),
        _element("MM-E3", "MULTIMODE_SOFTWARE", "MULTIMODE_SOFTWARE_ALGORITHM", "COMPUTE", "Opened task-local host compute", {key: opened[key] for key in ("candidate_runtime_seconds", "candidate_seconds_per_decode", "candidate_allocated_bytes_first_decode_max")}, claim_evidence("MM_OPENED_TASK_LOCAL_GAIN"), "Software runtime and sampled allocation, not RTL timing or physical power.", ["six-cycle latency", "FPGA resource"]),
        _element("MM-E4", "MULTIMODE_SOFTWARE", "MULTIMODE_SOFTWARE_ALGORITHM", "LER", "Phase-6D v1 strongest-baseline headroom", by_id["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_result"], claim_evidence("MM_V1_CAUSAL_HEADROOM_NO_GO"), "Mandatory negative: point and LCB are both zero.", ["remove denominator", "rescue with RTL"]),
        _element("MM-E5", "MULTIMODE_SOFTWARE", "MULTIMODE_SOFTWARE_ALGORITHM", "EVIDENCE_STATE", "Frozen-benchmark promotion state", None, claim_evidence("MM_FROZEN_BENCHMARK_SOTA_BLOCKED"), "T6.24.5 Dropped; pilot/formal unaccessed.", ["formal result", "frozen SOTA"]),
        _element("RTL-E1", "SINGLE_MODE_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "LATENCY_CYCLES", "Deterministic pipeline", {"cycles": 6, "II": 1}, claim_evidence("RTL_DETERMINISTIC_SIX_CYCLE_II1"), "Cycle contract, not physical nanoseconds.", ["multimode latency", "board latency"]),
        _element("RTL-E2", "SINGLE_MODE_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "PROPERTY", "Atomic/fail-closed property closure", by_id["RTL_ATOMIC_FAIL_CLOSED"]["current_result"], claim_evidence("RTL_ATOMIC_FAIL_CLOSED"), "Two-state pre-board property scope.", ["physical metastability proof", "multimode safety"]),
        _element("RTL-E3", "SINGLE_MODE_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "CXXRTL", "Million-cycle exact qualification", by_id["RTL_DETERMINISTIC_SIX_CYCLE_II1"]["current_result"], claim_evidence("RTL_DETERMINISTIC_SIX_CYCLE_II1"), "Full 148-byte public vector; zero mismatch.", ["four-state proof", "board correctness"]),
        _element("RTL-E4", "SINGLE_MODE_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "POST_ROUTE", "Three-seed whole-harness P&R", {key: postroute[key] for key in ("fmax_mhz", "critical_paths", "clock_model")}, claim_evidence("RTL_POST_ROUTE_ESTIMATE"), "Critical paths end in observability fold; conservative harness estimate.", ["bare-core Fmax", "fastest"]),
        _element("RTL-E5", "SINGLE_MODE_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "RESOURCE", "Three-seed resource range", postroute["resource_summary"], claim_evidence("RTL_POST_ROUTE_ESTIMATE"), "Post-route utilization range for the qualification harness.", ["board readback", "cross-device rank"]),
        _element("RTL-E6", "SINGLE_MODE_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "BOARD_NULL", "Physical board fields", by_id["RTL_BOARD_MEASUREMENT_BLOCKED"]["current_result"], claim_evidence("RTL_BOARD_MEASUREMENT_BLOCKED"), "All measured fields remain null.", ["measured latency", "measured power"]),
        _element("ML-E1", "LEARNING_EXTENSION", "LEARNED_APPROXIMATION_EXTENSION", "STATUS", "CNN/student disposition", by_id["LEARNING_APPROXIMATION_DROPPED"]["current_result"], claim_evidence("LEARNING_APPROXIMATION_DROPPED"), "Inset/status only; no primary result metric.", ["third primary lane", "algorithm or RTL promotion"]),
    ]
    edges = [
        {"edge_id": "MM-CONTEXT-TO-DECISION", "source_lane": "MULTIMODE_SOFTWARE_ALGORITHM", "target_lane": "MULTIMODE_SOFTWARE_ALGORITHM", "relation": "opened_context_then_preregistered_headroom_decision"},
        {"edge_id": "RTL-PROPERTY-TO-LONG", "source_lane": "SINGLE_MODE_DETERMINISTIC_RTL", "target_lane": "SINGLE_MODE_DETERMINISTIC_RTL", "relation": "same_exact_source_property_then_long_qualification"},
        {"edge_id": "RTL-LONG-TO-PR", "source_lane": "SINGLE_MODE_DETERMINISTIC_RTL", "target_lane": "SINGLE_MODE_DETERMINISTIC_RTL", "relation": "same_exact_source_long_qualification_then_postroute"},
    ]
    return {
        "figure_id": "PHASE6D_DUAL_LANE_MAIN_FIGURE",
        "panels": config["panels"],
        "elements": elements,
        "edges": edges,
        "global_weighted_score": None,
        "ranking_policy": "NO_CROSS_LANE_RANKING_OR_GATE_SUBSTITUTION",
        "caption_contract": "Panel A reports multimode software LER/tail/compute and the mandatory v1 NO-GO; Panel B reports exact single-mode RTL cycles/properties/CXXRTL/P&R with board-null; the learning inset reports Dropped/absent only.",
    }


def _forbidden_transfers() -> list[dict[str, str]]:
    rows = [
        ("MM_LER_TO_RTL", "MULTIMODE_SOFTWARE_ALGORITHM", "SINGLE_MODE_DETERMINISTIC_RTL", "attach multimode LER to current RTL implementation"),
        ("RTL_LATENCY_TO_MM", "SINGLE_MODE_DETERMINISTIC_RTL", "MULTIMODE_SOFTWARE_ALGORITHM", "attach six-cycle latency to the multimode software decoder"),
        ("MM_POSITIVE_TO_MM_SOTA", "MULTIMODE_SOFTWARE_ALGORITHM", "MULTIMODE_SOFTWARE_ALGORITHM", "promote opened T6.18.3 context to frozen strongest-baseline SOTA"),
        ("MM_NEGATIVE_RESOLVED_BY_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "MULTIMODE_SOFTWARE_ALGORITHM", "use RTL safety to rescue zero multimode headroom"),
        ("RTL_SAFETY_RESOLVED_BY_MM", "MULTIMODE_SOFTWARE_ALGORITHM", "SINGLE_MODE_DETERMINISTIC_RTL", "use LER evidence to satisfy RTL property gates"),
        ("LEARNING_TO_MM_PROMOTION", "LEARNED_APPROXIMATION_EXTENSION", "MULTIMODE_SOFTWARE_ALGORITHM", "use absent student as algorithm promotion evidence"),
        ("LEARNING_TO_RTL_PROMOTION", "LEARNED_APPROXIMATION_EXTENSION", "SINGLE_MODE_DETERMINISTIC_RTL", "use learned accuracy as atomic/fail-closed evidence"),
        ("PREBOARD_TO_BOARD", "SINGLE_MODE_DETERMINISTIC_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "rename P&R/clock model as measured"),
        ("HARNESS_FMAX_TO_BARE_CORE", "SINGLE_MODE_DETERMINISTIC_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "rename observability-harness Fmax as bare-core Fmax"),
        ("CROSS_LANE_WEIGHTED_SCORE", "MULTIMODE_SOFTWARE_ALGORITHM", "SINGLE_MODE_DETERMINISTIC_RTL", "combine LER, latency and safety into one weighted rank"),
    ]
    return [
        {"transfer_id": transfer_id, "source_lane": source, "target_lane": target, "trigger": trigger, "disposition": "REJECT"}
        for transfer_id, source, target, trigger in rows
    ]


def _parent_verification() -> dict[str, Any]:
    return {
        "dual_contract": dual_gate.verify_report(),
        "opened_multimode": opened_gate.verify_report(),
        "multimode_headroom": headroom_gate.verify(),
        "rtl_formal": formal_gate.verify(),
        "rtl_long": long_gate.verify(),
        "rtl_hardware": hardware_gate.verify(),
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(section: str, row_id: str, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        rows.append({
            "section": section,
            "row_id": row_id,
            "lane_or_panel": str(payload.get("lane_id", payload.get("source_lane", "META"))),
            "state_or_namespace": str(payload.get("state", payload.get("metric_namespace", payload.get("disposition", "BINDING")))),
            "payload_json": encoded,
            "payload_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        })

    for claim in report["claims"]:
        add("claim", str(claim["claim_id"]), claim)
    for element in report["figure_contract"]["elements"]:
        add("figure_element", str(element["element_id"]), element)
    for transfer in report["forbidden_transfers"]:
        add("forbidden_transfer", str(transfer["transfer_id"]), transfer)
    for key, binding in report["artifact_registry"].items():
        add("artifact", key, {"artifact_id": key, **binding})
    return rows


def _write_source_data(report: Mapping[str, Any]) -> int:
    rows = _source_rows(report)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _source_data_matches(report: Mapping[str, Any]) -> bool:
    try:
        with SOURCE_DATA.open("r", encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return False
    expected = _source_rows(report)
    return actual == expected and all(
        hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest() == row["payload_sha256"]
        for row in actual
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    claims = {row["claim_id"]: row for row in report["claims"]}
    opened = claims["MM_OPENED_TASK_LOCAL_GAIN"]["current_result"]
    no_go = claims["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_result"]
    rtl = claims["RTL_POST_ROUTE_ESTIMATE"]["current_result"]
    claim_rows = "\n".join(
        f"| `{row['claim_id']}` | `{row['lane_id']}` | `{row['state']}` | {row['boundary']} |"
        for row in report["claims"]
    )
    return f"""# T6.26.3 双证据 lane claim-evidence / 主图合同

## 结论

**`{report['verdict']}`**。论文必须保留两个并列但不可互相补门的 primary lane：

- multimode software：T6.18.3 的 opened task-local positive 结果为 `p_L={opened['candidate_p_L']:.6f}` vs static-Euclidean `{opened['static_euclidean_p_L']:.6f}`；但 Phase-6D v1 对 strongest retained static-mixture exact MLD 的 point/LCB 均为 `{no_go['relative_improvement_point']:.1%}/{no_go['relative_improvement_lcb']:.1%}`，因此 frozen-benchmark SOTA 未建立，pilot/formal 未访问。
- single-mode RTL：六周期、II=1、百万周期全公开向量零 mismatch、atomic/fail-closed property 与三 seed 27 MHz P&R 已在同一 exact top 上闭合；Fmax min/median/max=`{rtl['fmax_mhz']['minimum']:.3f}/{rtl['fmax_mhz']['median']:.3f}/{rtl['fmax_mhz']['maximum']:.3f}` MHz。
- CNN/student：T6.26.1--T6.26.2 Dropped，主证据中 absent；只能作为未来新路线的可替换近似，不是第三 primary lane。

## 原子 claim matrix

| Claim | Lane | State | Boundary |
| --- | --- | --- | --- |
{claim_rows}

## 主图合同

Panel A 只显示 multimode LER/tail/compute/evidence state，并同时显示 task-local positive 与 strongest-baseline NO-GO。Panel B 只显示 single-mode cycles/property/CXXRTL/post-route/resource/board-null。Learning 只作 Dropped/absent inset。禁止 global weighted score、跨 lane 箭头、用一条 lane 满足另一条 lane 的 gate。

三条 post-route critical path 都终止于 observability fold，因此 36.794 MHz 是 whole-harness conservative estimate，不是 bare-core 或真板速度。所有 board-measured 字段保持 null。
"""


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> list[dict[str, Any]]:
    config = _load(CONFIG)
    claims = {row["claim_id"]: row for row in report["claims"]}
    elements = report["figure_contract"]["elements"]
    panels = {row["panel_id"]: row for row in report["figure_contract"]["panels"]}
    artifacts = report["artifact_registry"]
    expected_statuses = config["required_task_statuses"]
    expected_verdicts = config["required_parent_verdicts"]
    actual_verdicts = report["parent_verdicts"]
    headroom = report["parent_summaries"]["multimode_headroom"]
    opened = report["parent_summaries"]["opened_multimode"]
    formal = report["parent_summaries"]["rtl_formal"]
    long = report["parent_summaries"]["rtl_long"]
    hardware = report["parent_summaries"]["rtl_hardware"]
    evidence_complete = all(
        set(row["evidence"]) == {*EVIDENCE_CATEGORIES, "selectors"}
        and all(row["evidence"][category] for category in EVIDENCE_CATEGORIES)
        and all(key in artifacts for category in EVIDENCE_CATEGORIES for key in row["evidence"][category])
        and bool(row["evidence"]["selectors"])
        for row in [*report["claims"], *elements]
    )
    gates = [
        ("identity_and_frozen_config", report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION and [row["claim_id"] for row in report["claims"]] == config["claim_ids"]),
        ("all_six_parent_verifiers_pass", set(report["parent_verification"]) == set(expected_verdicts) and all(bool(value) for value in report["parent_verification"].values())),
        ("parent_verdicts_exact", actual_verdicts == expected_verdicts),
        ("board_states_exact_and_phase6d_active_or_done", report["board_snapshot"]["statuses"] == expected_statuses and report["board_snapshot"] == _board_snapshot()),
        ("lane_outcomes_are_independent", report["lane_outcomes"] == {"MULTIMODE_SOFTWARE_ALGORITHM": "NO_GO_PRIMARY_WITH_OPENED_TASK_LOCAL_CONTEXT", "SINGLE_MODE_DETERMINISTIC_RTL": "GO_EXACT_TOP_PREBOARD_HARDWARE_LANE", "LEARNED_APPROXIMATION_EXTENSION": "DROPPED_ABSENT"}),
        ("claim_states_and_wording_are_closed", set(row["state"] for row in report["claims"]) <= CLAIM_STATES and all(row["wording_status"] == "SAFE" and row["safe_wording"] and row["boundary"] and row["forbidden_wording"] for row in report["claims"])),
        ("every_claim_and_figure_element_has_report_raw_config_code_source_hashes", evidence_complete),
        ("all_artifact_hash_bindings_are_live", all(len(row["sha256"]) == 64 and int(row["bytes"]) > 0 for row in artifacts.values()) and (not check_live_files or all(_live(row) for row in artifacts.values()))),
        ("opened_positive_is_task_local_and_recomputed", opened["verdict"] == "GO_POSTERIOR_WEIGHTED_CPD_DRIFT_GAIN" and claims["MM_OPENED_TASK_LOCAL_GAIN"]["state"] == "RESULTS_ONLY_NONRANKING" and claims["MM_OPENED_TASK_LOCAL_GAIN"]["current_result"]["cycles"] == 9_600_000 and claims["MM_OPENED_TASK_LOCAL_GAIN"]["current_result"]["candidate_p_L"] == opened["candidate_p_L"] and claims["MM_OPENED_TASK_LOCAL_GAIN"]["current_result"]["static_euclidean_p_L"] == opened["static_euclidean_p_L"]),
        ("multimode_headroom_no_go_and_strongest_denominator_visible", headroom["verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM" and headroom["strongest_baseline"] == "static_mixture_exact_mld" and headroom["baseline_p_L"] == headroom["proposed_p_L"] and headroom["relative_improvement_point"] == headroom["relative_improvement_lcb"] == 0.0 and claims["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_result"]["relative_improvement_point"] == 0.0),
        ("multimode_formal_sota_is_blocked_and_unaccessed", claims["MM_FROZEN_BENCHMARK_SOTA_BLOCKED"]["state"] == "BLOCKED_NOT_RUN" and claims["MM_FROZEN_BENCHMARK_SOTA_BLOCKED"]["current_result"] is None and headroom["formal_or_pilot_accessed"] is False and report["board_snapshot"]["statuses"]["T6.24.5"] == "Dropped"),
        ("rtl_six_cycle_ii1_and_million_cycle_zero_mismatch", hardware["cycles"] == 6 and hardware["ii"] == 1 and long["cycles"] == 1_000_000 and long["mismatches"] == 0 and long["ii1_input_pairs"] == long["ii1_output_pairs"] > 0 and claims["RTL_DETERMINISTIC_SIX_CYCLE_II1"]["current_result"]["latency_cycles"] == 6),
        ("rtl_atomic_fail_closed_property_and_mutations_pass", formal["gates"] == {"passed": 17, "total": 17} and formal["mutations"] == {"killed": 21, "total": 21, "minimum": 18} and long["gates"] == {"passed": 19, "total": 19} and claims["RTL_ATOMIC_FAIL_CLOSED"]["state"] == "CURRENT_RESTRICTED"),
        ("three_seed_postroute_is_exact_and_wrapper_caveat_preserved", hardware["seeds"] == [1, 7, 19] and hardware["all_timing_pass"] and hardware["minimum_fmax_mhz"] >= 27.0 and hardware["minimum_fmax_mhz"] == claims["RTL_POST_ROUTE_ESTIMATE"]["current_result"]["fmax_mhz"]["minimum"] and hardware["wrapper_may_dominate_all"] and all(row["wrapper_may_dominate"] for row in claims["RTL_POST_ROUTE_ESTIMATE"]["current_result"]["critical_paths"])),
        ("board_measurements_remain_null_and_speed_claim_prohibited", all(value is None for value in hardware["measured_fields"].values()) and all(value is None for value in claims["RTL_BOARD_MEASUREMENT_BLOCKED"]["current_result"].values()) and claims["RTL_SPEED_ADVANTAGE_PROHIBITED"]["state"] == "PROHIBITED_POSITIVE" and claims["RTL_SPEED_ADVANTAGE_PROHIBITED"]["current_result"] == {"fastest_or_sota": False}),
        ("learning_is_dropped_absent_and_nonprimary", claims["LEARNING_APPROXIMATION_DROPPED"]["state"] == "DROPPED_ABSENT" and report["board_snapshot"]["statuses"]["T6.26.1"] == report["board_snapshot"]["statuses"]["T6.26.2"] == "Dropped" and claims["LEARNING_APPROXIMATION_DROPPED"]["current_result"]["present_in_primary_rtl"] is False),
        ("panel_metric_namespaces_and_wording_are_disjoint", all(element["panel_id"] in panels and element["lane_id"] == panels[element["panel_id"]]["lane_id"] and element["metric_namespace"] in panels[element["panel_id"]]["allowed_metric_namespaces"] and bool(element["allowed_wording"]) and bool(element["forbidden_interpretation"]) for element in elements)),
        ("no_cross_lane_edges_or_weighted_score", report["figure_contract"]["global_weighted_score"] is None and report["figure_contract"]["ranking_policy"] == "NO_CROSS_LANE_RANKING_OR_GATE_SUBSTITUTION" and all(edge["source_lane"] == edge["target_lane"] for edge in report["figure_contract"]["edges"])),
        ("all_ten_forbidden_transfers_are_explicit_rejections", [row["transfer_id"] for row in report["forbidden_transfers"]] == config["forbidden_transfer_ids"] and all(row["disposition"] == "REJECT" for row in report["forbidden_transfers"])),
        ("lossless_source_data_is_bound_and_reconstructed", report["source_data"]["rows"] == len(_source_rows(report)) and len(report["source_data"]["sha256"]) == 64 and (not check_live_files or (_live(report["source_data"]) and _source_data_matches(report)))),
    ]
    return [{"gate": name, "passed": bool(passed)} for name, passed in gates]


def semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def attempt(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(report)
        mutate(candidate)
        rejected = not all(row["passed"] for row in evaluate_gates(candidate, check_live_files=False))
        rows.append({"mutation": name, "rejected": rejected})

    def claim(candidate: dict[str, Any], claim_id: str) -> dict[str, Any]:
        return next(row for row in candidate["claims"] if row["claim_id"] == claim_id)

    def element(candidate: dict[str, Any], element_id: str) -> dict[str, Any]:
        return next(row for row in candidate["figure_contract"]["elements"] if row["element_id"] == element_id)

    attempt("promote_multimode_sota", lambda x: claim(x, "MM_FROZEN_BENCHMARK_SOTA_BLOCKED").update(state="CURRENT_RESTRICTED"))
    attempt("forge_headroom_gain", lambda x: x["parent_summaries"]["multimode_headroom"].update(relative_improvement_point=0.2))
    attempt("delete_strongest_denominator", lambda x: x["parent_summaries"]["multimode_headroom"].update(strongest_baseline="current_adaptive_weighted_cpd"))
    attempt("invent_formal_access", lambda x: x["parent_summaries"]["multimode_headroom"].update(formal_or_pilot_accessed=True))
    attempt("attach_rtl_latency_to_mm", lambda x: element(x, "MM-E1").update(metric_namespace="LATENCY_CYCLES"))
    attempt("attach_ler_to_rtl", lambda x: element(x, "RTL-E1").update(metric_namespace="LER"))
    attempt("break_formal_gate", lambda x: x["parent_summaries"]["rtl_formal"].update(gates={"passed": 16, "total": 17}))
    attempt("invent_cxxrtl_mismatch", lambda x: x["parent_summaries"]["rtl_long"].update(mismatches=1))
    attempt("change_latency_cycles", lambda x: x["parent_summaries"]["rtl_hardware"].update(cycles=5))
    attempt("forge_fmax", lambda x: x["parent_summaries"]["rtl_hardware"].update(minimum_fmax_mhz=999.0))
    attempt("erase_wrapper_caveat", lambda x: x["parent_summaries"]["rtl_hardware"].update(wrapper_may_dominate_all=False))
    attempt("invent_board_measurement", lambda x: x["parent_summaries"]["rtl_hardware"]["measured_fields"].update(board_latency_ns=1.0))
    attempt("promote_fastest", lambda x: claim(x, "RTL_SPEED_ADVANTAGE_PROHIBITED").update(state="CURRENT_RESTRICTED"))
    attempt("mark_learning_done", lambda x: x["board_snapshot"]["statuses"].update({"T6.26.2": "Done"}))
    attempt("make_learning_primary", lambda x: x["lane_outcomes"].update({"LEARNED_APPROXIMATION_EXTENSION": "GO_PRIMARY"}))
    attempt("add_global_weighted_score", lambda x: x["figure_contract"].update(global_weighted_score=1.0))
    attempt("insert_cross_lane_edge", lambda x: x["figure_contract"]["edges"][0].update(target_lane="SINGLE_MODE_DETERMINISTIC_RTL"))
    attempt("corrupt_artifact_hash", lambda x: x["artifact_registry"]["implementation"].update(sha256="0"))
    attempt("remove_element_config_evidence", lambda x: element(x, "MM-E1")["evidence"].update(configs=[]))
    attempt("erase_element_allowed_wording", lambda x: element(x, "RTL-E4").update(allowed_wording=""))
    attempt("forge_source_data_rows", lambda x: x["source_data"].update(rows=x["source_data"]["rows"] - 1))
    return {"detected": sum(int(row["rejected"]) for row in rows), "total": len(rows), "mutations": rows}


def _parent_summaries(
    opened: Mapping[str, Any], headroom: Mapping[str, Any], formal: Mapping[str, Any],
    long: Mapping[str, Any], hardware: Mapping[str, Any],
) -> dict[str, Any]:
    opened_aggregate = opened["summaries"]["aggregate"]
    boot = headroom["paired_bootstrap"]
    return {
        "opened_multimode": {
            "verdict": opened["verdict"],
            "candidate_p_L": opened_aggregate["observed_only_posterior_predictive_weighted"]["p_L"],
            "static_euclidean_p_L": opened_aggregate["static_euclidean"]["p_L"],
            "cycles": opened["formal_counts"]["total_physical_cycles"],
            "claim_boundary": opened["claim_boundary"],
        },
        "multimode_headroom": {
            "verdict": headroom["verdict"],
            "strongest_baseline": headroom["strongest_development_baseline_selection"]["selected"],
            "baseline_p_L": boot["baseline_p_L"],
            "proposed_p_L": boot["proposed_p_L"],
            "relative_improvement_point": boot["relative_improvement_point"],
            "relative_improvement_lcb": boot["relative_improvement_lcb"],
            "formal_or_pilot_accessed": headroom["scope"]["formal_or_pilot_accessed"],
        },
        "rtl_formal": {
            "verdict": formal["verdict"], "gates": formal["gate_summary"],
            "mutations": formal["mutation_summary"],
        },
        "rtl_long": {
            "verdict": long["verdict"], "gates": long["gate_summary"],
            "cycles": long["aggregate_python"]["cycles"],
            "mismatches": sum(int(row["mismatches"]) for row in long["cxxrtl_families"]),
            "ii1_input_pairs": long["aggregate_python"]["ii1_input_pairs"],
            "ii1_output_pairs": long["aggregate_python"]["ii1_output_pairs"],
        },
        "rtl_hardware": {
            "verdict": hardware["verdict"], "gates": hardware["gate_summary"],
            "cycles": hardware["clock_model"]["cycles"],
            "ii": hardware["clock_model"]["initiation_interval_cycles"],
            "seeds": [row["seed"] for row in hardware["place_route"]],
            "all_timing_pass": all(row["timing_pass"] for row in hardware["place_route"]),
            "minimum_fmax_mhz": hardware["fmax_mhz"]["minimum"],
            "wrapper_may_dominate_all": all(row["wrapper_may_dominate"] for row in hardware["critical_paths"]),
            "measured_fields": hardware["measured_fields"],
        },
    }


def build_report() -> dict[str, Any]:
    config = _load(CONFIG)
    opened = _load(OPENED_REPORT)
    headroom = _load(HEADROOM_REPORT)
    formal = _load(FORMAL_REPORT)
    long = _load(LONG_REPORT)
    hardware = _load(HARDWARE_REPORT)
    parent_checks = _parent_verification()
    claims = _claims(opened, headroom, formal, long, hardware)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_verification": {key: True for key in parent_checks},
        "parent_verdicts": {
            "dual_contract": _load(DUAL_REPORT)["verdict"],
            "opened_multimode": opened["verdict"],
            "multimode_headroom": headroom["verdict"],
            "rtl_formal": formal["verdict"],
            "rtl_long": long["verdict"],
            "rtl_hardware": hardware["verdict"],
        },
        "parent_summaries": _parent_summaries(opened, headroom, formal, long, hardware),
        "board_snapshot": _board_snapshot(),
        "lane_outcomes": {
            "MULTIMODE_SOFTWARE_ALGORITHM": "NO_GO_PRIMARY_WITH_OPENED_TASK_LOCAL_CONTEXT",
            "SINGLE_MODE_DETERMINISTIC_RTL": "GO_EXACT_TOP_PREBOARD_HARDWARE_LANE",
            "LEARNED_APPROXIMATION_EXTENSION": "DROPPED_ABSENT",
        },
        "artifact_registry": {key: _binding(path) for key, path in ARTIFACT_PATHS.items()},
        "claims": claims,
        "figure_contract": _figure_contract(claims),
        "forbidden_transfers": _forbidden_transfers(),
        "evidence_boundary": {
            "global_weighted_score": "PROHIBITED",
            "cross_lane_gate_substitution": "PROHIBITED",
            "multimode_frozen_benchmark_sota": False,
            "single_mode_preboard_hardware_lane": True,
            "board_measured": False,
            "fastest_or_sota_hardware": False,
            "multimode_decoder_in_rtl": False,
            "learning_primary": False,
        },
    }
    rows = _write_source_data(report)
    report["source_data"] = {**_binding(SOURCE_DATA), "rows": rows}
    _atomic_text(MARKDOWN, _render_markdown({**report, "verdict": VERDICT}))
    report["markdown"] = _binding(MARKDOWN)
    report["gates"] = evaluate_gates(report)
    audit = semantic_mutation_audit(report)
    report["semantic_mutations"] = {"detected": audit["detected"], "total": audit["total"]}
    report["semantic_mutation_results"] = audit["mutations"]
    report["gates"].append({
        "gate": "all_twenty_one_semantic_mutations_rejected",
        "passed": audit["detected"] == audit["total"] == int(config["semantic_mutation_count"]),
    })
    report["gate_summary"] = {
        "passed": sum(int(row["passed"]) for row in report["gates"]),
        "total": len(report["gates"]),
    }
    report["verdict"] = VERDICT if all(row["passed"] for row in report["gates"]) else "FAIL_CLOSED_DUAL_LANE_EVIDENCE_MATRIX"
    canonical = copy.deepcopy(report)
    canonical.pop("generated_at_utc", None)
    report["analysis_sha256"] = _canonical_sha256(canonical)
    return report


def _validate(report: Mapping[str, Any], *, check_live_files: bool = True) -> None:
    _require(report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION, "identity drift")
    _require(report["verdict"] == VERDICT, "wrong verdict")
    _require(report["gate_summary"] == {"passed": 21, "total": 21}, "gate closure failed")
    _require(report["semantic_mutations"] == {"detected": 21, "total": 21}, "mutation closure failed")
    _require(report["gates"][:-1] == evaluate_gates(report, check_live_files=check_live_files), "stored gate mismatch")
    recomputed = semantic_mutation_audit(report)
    _require(report["semantic_mutation_results"] == recomputed["mutations"], "stored mutation mismatch")
    _require(all(row["passed"] for row in report["gates"]), "failed gate")
    if check_live_files:
        _require(_live(report["source_data"]), "source data binding mismatch")
        _require(_live(report["markdown"]), "markdown binding mismatch")


def verify() -> dict[str, Any]:
    report = _load(REPORT)
    _validate(report)
    canonical = copy.deepcopy(report)
    expected = canonical.pop("analysis_sha256")
    canonical.pop("generated_at_utc", None)
    _require(_canonical_sha256(canonical) == expected, "analysis hash mismatch")
    return {
        "verdict": report["verdict"],
        "gates": report["gate_summary"],
        "mutations": report["semantic_mutations"],
        "analysis_sha256": expected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return 0
    report = build_report()
    _atomic_text(REPORT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["verdict"] == VERDICT:
        _validate(report)
    print(json.dumps({
        "verdict": report["verdict"],
        "gates": report["gate_summary"],
        "mutations": report["semantic_mutations"],
        "lane_outcomes": report["lane_outcomes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
