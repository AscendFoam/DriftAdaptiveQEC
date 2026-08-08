"""Build the T7.3.3 experiment-relevance reviewer response and evidence ladder."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.3.3"
SCHEMA_VERSION = "t7.3.3-experiment-relevance-reviewer-contract-v1"
VERDICT = "PASS_EXPERIMENT_RELEVANCE_WITHOUT_HARDWARE_OVERCLAIM"

CONFIG = ROOT / "configs/phase6d/t7_3_3_experiment_relevance_reviewer_contract.json"
BOARD = ROOT / "docs/new_task_board.md"
RISKS = ROOT / "docs/new_risks.md"
EXPERIMENT_PLAN = ROOT / "docs/experiment_plan.md"
MANUSCRIPT = ROOT / "docs/paper_notes/Phase6D_Dual_Lane_GKP_manuscript.tex"
MANUSCRIPT_CONTRACT = ROOT / "docs/t7_2_6_phase6d_manuscript_delta.json"
ATLAS = ROOT / "docs/t6_19_3_secondary_evidence_integrity.json"
AQEC = ROOT / "docs/t6_18_1_aqec_common_wallclock_replay.json"
HEADROOM = ROOT / "docs/t6_20_4_multimode_causal_headroom.json"
FORMAL = ROOT / "docs/t6_25_2_converged_rtl_formal.json"
LONG_RTL = ROOT / "docs/t6_25_3_converged_long_rtl.json"
HARDWARE = ROOT / "docs/t6_25_4_converged_hardware.json"
BOARD_BLOCKER = ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json"
PREBOARD = ROOT / "docs/t6_9_2_preboard_bitstream_candidate.json"
HIL_AUDIT = ROOT / "docs/03_hil_p4_boundary_audit.md"
FINAL_GATE = ROOT / "docs/t6_26_4_final_dual_lane_gate.json"

DEFAULT_REPORT = ROOT / "docs/t7_3_3_experiment_relevance_reviewer_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_3_3_experiment_relevance_reviewer_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/experiment_relevance_reviewer_response.md"

RESPONSE_STATES = {
    "REVIEWER_CONCERN",
    "TERM_BOUNDARY",
    "LITERATURE_FACT",
    "OFFICIAL_CODE_REPRODUCTION",
    "PROJECT_NATIVE_SIMULATION",
    "MOCK_SOFTWARE_HIL",
    "PREBOARD_RTL",
    "PHYSICAL_BOARD",
    "QUANTUM_HARDWARE",
    "NONTRANSFER",
    "MANUSCRIPT_CHANGE",
    "FUTURE_PROMOTION",
    "RESPONSE_WORDING",
    "RISK_DISCLOSURE",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    if binding.get("binding_kind") == "semantic_projection":
        if not path.is_file():
            return False
        projection = _governance_projection(
            binding["selector"], path.read_text(encoding="utf-8")
        )
        return projection == binding.get("projection") and _canonical_sha256(projection) == binding["sha256"]
    return path.exists() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _task_status(board: str, task_id: str) -> str:
    match = re.search(rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|", board, re.MULTILINE)
    if not match:
        raise ValueError(f"task status not found: {task_id}")
    return match.group(1).strip()


def _governance_projection(selector: str, text: str) -> dict[str, Any]:
    if selector == "t7.3.3_board":
        tasks = (
            "T7.3.3", "T7.3.4", "T8.1.1", "T8.1.2", "T8.1.3",
            "T8.2.1", "T8.2.2", "T8.2.3",
        )
        return {"statuses": {task: _task_status(text, task) for task in tasks}}
    if selector == "t7.3.3_risks":
        return {
            "r_n160_present": "R-N160" in text,
            "task_audit_present": "| 2026-07-21 | T7.3.3 |" in text,
        }
    if selector == "t7.3.3_plan":
        return {
            "offline_before_closed_loop": "优先做**离线 re-decoding**，再做闭环" in text,
        }
    if selector == "t7.3.3_manuscript_contract":
        payload = json.loads(text)
        return {
            key: payload[key]
            for key in ("task_id", "schema_version", "verdict", "gate_summary", "analysis_sha256")
        }
    raise ValueError(f"unknown governance selector: {selector}")


def _semantic_binding(path: Path, selector: str, text: str) -> dict[str, Any]:
    projection = _governance_projection(selector, text)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "binding_kind": "semantic_projection",
        "selector": selector,
        "projection": projection,
        "sha256": _canonical_sha256(projection),
    }


def _tex_title_and_abstract(text: str) -> tuple[str, str]:
    title = re.search(r"\\title\{(.*?)\}\s*\\author", text, re.DOTALL)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
    if not title or not abstract:
        raise ValueError("canonical manuscript title/abstract not found")
    return re.sub(r"\s+", " ", title.group(1)), re.sub(r"\s+", " ", abstract.group(1))


def _response_text() -> str:
    return (
        "We agree that the phrase experiment-related is ambiguous when no quantum processor or physical FPGA measurement is part of the study. "
        "We therefore use experiment-informed or experiment-facing, always paired with the evidence modality, and do not describe this work as experimental GKP quantum error correction.\n\n"
        "The revised evidence ladder separates seven levels. Literature-reported physical results, including AQEC lifetime gains and Sivak-style break-even, remain facts about the cited systems. "
        "Two official-code CPD threshold cells are numerical reproductions in a different surface-lattice task. Our own decoder and AQEC results are project-native simulations; the legacy HIL path uses a mock FPGA backend. "
        "None of those levels is a board or quantum-hardware measurement.\n\n"
        "The hardware contribution is narrower but substantive. The exact single-mode production top passes 17 formal gates, kills 21 targeted mutants, and matches an independent reference over ten 100,000-cycle CXXRTL families with zero full-vector mismatch, undefined action or silent overflow. "
        "Three open-source place-and-route seeds meet the 27-MHz contract, with whole-harness Fmax between 36.794 and 37.869 MHz. These are pre-board digital qualification results, not measured latency, jitter, deadline or power.\n\n"
        "The dedicated board gate remains blocked: every physical measurement field is null, the historical UART candidate was neither programmed nor measured, and the optional real-GKP-data and control-chain tasks remain Todo. "
        "Accordingly, AQEC or Sivak evidence cannot validate our simulator, decoder, RTL or board. The present experiment-facing relevance comes from causal observation contracts, fault paths, deadlines and deployment interfaces designed for a future control chain, not from claiming that such an experiment has already been performed.\n\n"
        "We have made this distinction explicit in the title, abstract, Methods, Results and Limitations. A future promotion to physical evidence will require a board-identified, bitstream-bound measurement pack and, separately, licensed real GKP syndrome data with protocol metadata and valid labels or tomography."
    )


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"generated_at_utc", "analysis_sha256", "semantic_mutation_audit", "source_data", "markdown"}
    return {key: value for key, value in report.items() if key not in omitted}


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in report["response_rows"]:
        rows.append({
            "row_id": row["row_id"],
            "response_state": row["response_state"],
            "topic": row["topic"],
            "claim": row["claim"],
            "boundary": row["boundary"],
            "source_ids_json": _canonical(row["source_ids"]),
            "row_sha256": _canonical_sha256(row),
        })
    return rows


def _source_data_matches(report: Mapping[str, Any], path: Path = DEFAULT_SOURCE_DATA) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream)) == _source_rows(report)


def build_report(*, generated_at_utc: str | None = None) -> dict[str, Any]:
    config = _load(CONFIG)
    board_text = BOARD.read_text(encoding="utf-8")
    risk_text = RISKS.read_text(encoding="utf-8")
    plan_text = EXPERIMENT_PLAN.read_text(encoding="utf-8")
    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    hil_text = HIL_AUDIT.read_text(encoding="utf-8")
    atlas = _load(ATLAS)
    aqec = _load(AQEC)
    headroom = _load(HEADROOM)
    formal = _load(FORMAL)
    long_rtl = _load(LONG_RTL)
    hardware = _load(HARDWARE)
    blocker = _load(BOARD_BLOCKER)
    preboard = _load(PREBOARD)
    final_gate = _load(FINAL_GATE)

    cells = atlas["cells"]
    value_counts = dict(sorted(Counter(cell["value_state"] for cell in cells).items()))
    grade_counts = dict(sorted(Counter(cell["evidence_grade"] for cell in cells).items()))
    official_cells = [cell for cell in cells if cell["evidence_grade"] == "OFFICIAL_CODE_REPRODUCTION"]
    title, abstract = _tex_title_and_abstract(manuscript_text)
    title_abstract = f"{title} {abstract}".lower()
    measured = blocker["measured_results"]
    hardware_measured = hardware["measured_fields"]
    aggregate = long_rtl["aggregate_python"]

    artifact_paths = {
        "implementation": Path(__file__).resolve(),
        "config": CONFIG,
        "manuscript": MANUSCRIPT,
        "atlas": ATLAS,
        "aqec": AQEC,
        "headroom": HEADROOM,
        "formal": FORMAL,
        "long_rtl": LONG_RTL,
        "hardware": HARDWARE,
        "board_blocker": BOARD_BLOCKER,
        "preboard": PREBOARD,
        "hil_audit": HIL_AUDIT,
        "final_gate": FINAL_GATE,
    }
    artifact_registry = {key: _binding(path) for key, path in artifact_paths.items()}
    artifact_registry.update({
        "task_board": _semantic_binding(BOARD, "t7.3.3_board", board_text),
        "new_risks": _semantic_binding(RISKS, "t7.3.3_risks", risk_text),
        "experiment_plan": _semantic_binding(EXPERIMENT_PLAN, "t7.3.3_plan", plan_text),
        "manuscript_contract": _semantic_binding(
            MANUSCRIPT_CONTRACT,
            "t7.3.3_manuscript_contract",
            MANUSCRIPT_CONTRACT.read_text(encoding="utf-8"),
        ),
    })

    ladder = [
        {"level":"LITERATURE_FACT","status":"AVAILABLE_CONTEXT_ONLY","allowed":"attribute reported values to the cited physical system","forbidden":"project measurement or reproduction","evidence":{"literature_value_cells":value_counts.get("LITERATURE_VALUE",0),"literature_only_cells":grade_counts.get("LITERATURE_ONLY",0),"null_not_reported_cells":value_counts.get("NULL_NOT_REPORTED",0)}},
        {"level":"OFFICIAL_CODE_REPRODUCTION","status":"AVAILABLE_TWO_CPD_CELLS","allowed":"report source-qualified numerical reproduction within its task signature","forbidden":"device or project-native hardware evidence","evidence":{"count":len(official_cells),"cell_ids":[cell["cell_id"] for cell in official_cells],"values":[cell["value"] for cell in official_cells],"physical_measurement":False}},
        {"level":"PROJECT_NATIVE_SIMULATION","status":"AVAILABLE_MIXED_POSITIVE_NEGATIVE","allowed":"report simulator-scoped LER, lifetime and fault-path results","forbidden":"physical lifetime, device uncertainty or break-even","evidence":{"aqec_verdict":aqec["verdict"],"aqec_seed_clusters":aqec["project_result"]["seed_clusters"],"aqec_source_rows":aqec["source_data"]["rows"],"multimode_verdict":headroom["verdict"],"multimode_relative_improvement":headroom["paired_bootstrap"]["relative_improvement_point"]}},
        {"level":"MOCK_SOFTWARE_HIL","status":"AVAILABLE_LEGACY_MOCK_ONLY","allowed":"report software orchestration and mock event semantics","forbidden":"real-board HIL or physical timing","evidence":{"software_orchestrator":"software_hil_orchestrator" in hil_text,"mock_backend":"mock backend" in hil_text,"placeholder_board_backend":"placeholder_real_board_backend" in hil_text,"real_board_hil":False}},
        {"level":"PREBOARD_DIGITAL_QUALIFICATION","status":"AVAILABLE_EXACT_SINGLE_MODE","allowed":"formal/CXXRTL/P&R-qualified deterministic atomic fail-closed architecture","forbidden":"measured board latency/power, multimode RTL or fastest FPGA","evidence":{"formal_gates":formal["gate_summary"]["passed"],"formal_mutants":formal["mutation_summary"]["killed"],"cxxrtl_cycles":aggregate["cycles"],"latency_violations":aggregate["latency_violations"],"undefined_actions":aggregate["undefined_actions"],"silent_overflow":aggregate["silent_overflow"],"fmax_min_mhz":hardware["fmax_mhz"]["minimum"],"fmax_max_mhz":hardware["fmax_mhz"]["maximum"],"place_route_seeds":len(hardware["place_route"]),"measured":False}},
        {"level":"PHYSICAL_BOARD_MEASUREMENT","status":"BLOCKED_ALL_FIELDS_NULL","allowed":"state blocker and recovery conditions","forbidden":"board correctness, latency, jitter, deadline, resources, power or speed","evidence":{"verdict":blocker["verdict"],"field_count":len(measured),"null_count":sum(value is None for value in measured.values()),"all_null":all(value is None for value in measured.values()),"current_hardware_measured_all_null":all(value is None for value in hardware_measured.values()),"historical_candidate_programmed":preboard["board_programmed"],"historical_measurements_collected":preboard["physical_measurements_collected"]}},
        {"level":"QUANTUM_HARDWARE_OR_REAL_GKP_DATA","status":"ABSENT_OPTIONAL_PHASE8_TODO","allowed":"state optional future route and required permissions/metadata","forbidden":"cavity/transmon, real-syndrome, frame-update or active-feedback result","evidence":{"phase8_statuses":{task:_task_status(board_text,task) for task in ("T8.1.1","T8.1.2","T8.1.3","T8.2.1","T8.2.2","T8.2.3")},"real_gkp_data":False,"quantum_control_chain":False}},
    ]

    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "reviewer_context": config["reviewer_context"],
        "task_status": {task:_task_status(board_text,task) for task in ("T7.3.3","T7.3.4")},
        "terminology_contract": {
            "preferred_terms": config["preferred_terms"],
            "forbidden_title_abstract_phrases": config["forbidden_title_abstract_phrases"],
            "experimental_gkp_qec_claim": False,
            "experiment_facing_means_physical_execution": False,
        },
        "evidence_ladder": ladder,
        "nontransfer_contract": {
            "aqec_or_sivak_to_project_physical_evidence": False,
            "official_code_to_physical_evidence": False,
            "mock_hil_to_board_measurement": False,
            "rtl_property_to_quantum_correction": False,
            "preboard_estimate_to_measurement": False,
        },
        "manuscript_audit": {
            "title": title,
            "abstract": abstract,
            "forbidden_phrase_presence": {phrase: phrase.lower() in title_abstract for phrase in config["forbidden_title_abstract_phrases"]},
            "required_markers": {marker: marker in re.sub(r"\s+", " ", manuscript_text) for marker in config["required_manuscript_markers"]},
            "named_locations": ["Title", "Abstract", "Methods", "Results", "Limitations", "Conclusion"],
        },
        "external_validity": {
            "resolved": False,
            "current_limit": "no physical FPGA measurement, real GKP syndrome dataset, cavity/transmon control chain or active feedback",
            "phase8_optional": "## Phase 8：可选真实 GKP 数据或量子硬件接入" in board_text,
            "plan_requires_offline_before_closed_loop": "优先做**离线 re-decoding**，再做闭环" in plan_text,
        },
        "response_package": {
            "strategy": {
                "overall_posture": "accept ambiguous terminology; replace umbrella experimental wording with a seven-level evidence ladder",
                "major_risks": ["literature-to-project transfer", "reproduction-to-measurement transfer", "mock-HIL-to-board transfer", "RTL-to-QPU transfer"],
                "suggested_order": ["direct concession", "evidence ladder", "digital contribution", "absent physical evidence", "future promotion route"],
            },
            "tracker": {
                "comment_id": config["reviewer_context"]["comment_id"],
                "concern": config["reviewer_context"]["reviewer_concern"],
                "category": config["reviewer_context"]["category"],
                "severity": config["reviewer_context"]["severity"],
                "actions": config["reviewer_context"]["actions"],
                "manuscript_locations": ["Title", "Abstract", "Methods: Evidence modalities", "Results", "Limitations", "Conclusion"],
                "missing_author_input": config["reviewer_context"]["visible_placeholder"],
            },
            "english_response": _response_text(),
            "manuscript_change_checklist": [
                "Use experiment-informed/facing only with an adjacent evidence modality.",
                "Keep the title free of experimental-GKP or device-demonstration language.",
                "Keep simulation, mock HIL, formal/CXXRTL, P&R estimate and physical measurement in separate rows.",
                "Keep every physical-board field null until a board-identified raw measurement pack exists.",
                "Keep AQEC/Sivak physical results attributed to their systems and optional Phase 8 explicitly future-facing.",
            ],
            "missing_information": [config["reviewer_context"]["visible_placeholder"]],
            "package_readiness": config["reviewer_context"]["package_readiness"],
        },
        "response_rows": config["response_rows"],
        "forbidden_response_phrases": config["forbidden_response_phrases"],
        "artifact_registry": artifact_registry,
        "risk_audit": {"r_n160_present":"R-N160" in risk_text,"t7_3_3_audit_present":"| 2026-07-21 | T7.3.3 |" in risk_text},
        "final_gate_context": {
            "verdict": final_gate["verdict"],
            "truth_key": final_gate["truth_key"],
            "board_claim": "pre-board deterministic atomic fail-closed" if final_gate["publication_boundary"]["single_mode_preboard_deterministic_atomic_fail_closed"] else "not qualified",
            "board_measured": final_gate["publication_boundary"]["board_measured"],
        },
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {"passed":sum(report["gates"].values()),"total":len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_EXPERIMENT_RELEVANCE_CONTRACT"
    report["semantic_mutation_audit"] = _run_mutations(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    ladder = report["evidence_ladder"]
    by_level = {row["level"]:row for row in ladder}
    rows = report["response_rows"]
    registry = report["artifact_registry"]
    allowed_sources = set(registry)
    text = report["response_package"]["english_response"].lower()
    gates = {
        "G01_identity_and_task_handoff_are_exact": report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION and report["task_status"]["T7.3.3"] == "Done" and report["task_status"]["T7.3.4"] in {"In Progress", "Done"},
        "G02_preemptive_context_and_placeholder_are_honest": report["reviewer_context"]["comment_id"] == "PRQ-HW-1" and report["response_package"]["package_readiness"] == "draft_with_placeholders" and report["response_package"]["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"],
        "G03_evidence_ladder_is_ordered_and_noncollapsed": [row["level"] for row in ladder] == ["LITERATURE_FACT","OFFICIAL_CODE_REPRODUCTION","PROJECT_NATIVE_SIMULATION","MOCK_SOFTWARE_HIL","PREBOARD_DIGITAL_QUALIFICATION","PHYSICAL_BOARD_MEASUREMENT","QUANTUM_HARDWARE_OR_REAL_GKP_DATA"],
        "G04_literature_values_remain_context_only": by_level["LITERATURE_FACT"]["evidence"] == {"literature_value_cells":57,"literature_only_cells":162,"null_not_reported_cells":107} and by_level["LITERATURE_FACT"]["forbidden"] == "project measurement or reproduction",
        "G05_official_code_reproduction_is_numerical_not_physical": by_level["OFFICIAL_CODE_REPRODUCTION"]["evidence"]["count"] == 2 and set(by_level["OFFICIAL_CODE_REPRODUCTION"]["evidence"]["cell_ids"]) == {"structured_official_cpd_threshold","structured_official_analog_mwpm_threshold"} and by_level["OFFICIAL_CODE_REPRODUCTION"]["evidence"]["physical_measurement"] is False,
        "G06_aqec_project_replay_is_deep_but_not_paper_native": by_level["PROJECT_NATIVE_SIMULATION"]["evidence"]["aqec_verdict"] == "PASS_PROJECT_NATIVE_AQEC_WALLCLOCK_WITH_OFFICIAL_PROTOCOL_BLOCKED" and by_level["PROJECT_NATIVE_SIMULATION"]["evidence"]["aqec_seed_clusters"] == 144 and by_level["PROJECT_NATIVE_SIMULATION"]["evidence"]["aqec_source_rows"] == 144152,
        "G07_multimode_simulation_negative_result_stays_visible": by_level["PROJECT_NATIVE_SIMULATION"]["evidence"]["multimode_verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM" and by_level["PROJECT_NATIVE_SIMULATION"]["evidence"]["multimode_relative_improvement"] == 0.0,
        "G08_software_hil_is_explicitly_mock_not_board": all(by_level["MOCK_SOFTWARE_HIL"]["evidence"][key] for key in ("software_orchestrator","mock_backend","placeholder_board_backend")) and by_level["MOCK_SOFTWARE_HIL"]["evidence"]["real_board_hil"] is False,
        "G09_formal_property_and_mutation_counts_are_exact": by_level["PREBOARD_DIGITAL_QUALIFICATION"]["evidence"]["formal_gates"] == 17 and by_level["PREBOARD_DIGITAL_QUALIFICATION"]["evidence"]["formal_mutants"] == 21,
        "G10_long_rtl_is_million_cycle_full_safety_replay": by_level["PREBOARD_DIGITAL_QUALIFICATION"]["evidence"]["cxxrtl_cycles"] == 1_000_000 and all(by_level["PREBOARD_DIGITAL_QUALIFICATION"]["evidence"][key] == 0 for key in ("latency_violations","undefined_actions","silent_overflow")),
        "G11_place_route_is_three_seed_preboard_estimate": by_level["PREBOARD_DIGITAL_QUALIFICATION"]["evidence"]["place_route_seeds"] == 3 and by_level["PREBOARD_DIGITAL_QUALIFICATION"]["evidence"]["fmax_min_mhz"] >= 36.79 and by_level["PREBOARD_DIGITAL_QUALIFICATION"]["evidence"]["measured"] is False,
        "G12_physical_board_gate_is_blocked_and_all_null": by_level["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]["verdict"] == "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL" and by_level["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]["all_null"] and by_level["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]["field_count"] == by_level["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]["null_count"] and by_level["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]["current_hardware_measured_all_null"] and by_level["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]["historical_candidate_programmed"] is False and by_level["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]["historical_measurements_collected"] is False,
        "G13_real_gkp_and_control_chain_remain_optional_todo": set(by_level["QUANTUM_HARDWARE_OR_REAL_GKP_DATA"]["evidence"]["phase8_statuses"].values()) == {"Todo"} and by_level["QUANTUM_HARDWARE_OR_REAL_GKP_DATA"]["evidence"]["real_gkp_data"] is False and by_level["QUANTUM_HARDWARE_OR_REAL_GKP_DATA"]["evidence"]["quantum_control_chain"] is False,
        "G14_all_cross_level_evidence_transfers_are_forbidden": report["nontransfer_contract"] == {"aqec_or_sivak_to_project_physical_evidence":False,"official_code_to_physical_evidence":False,"mock_hil_to_board_measurement":False,"rtl_property_to_quantum_correction":False,"preboard_estimate_to_measurement":False},
        "G15_title_and_abstract_avoid_experimental_gkp_claim": report["terminology_contract"]["experimental_gkp_qec_claim"] is False and report["terminology_contract"]["experiment_facing_means_physical_execution"] is False and not any(report["manuscript_audit"]["forbidden_phrase_presence"].values()),
        "G16_manuscript_discloses_preboard_null_and_physical_lifetime_boundaries": all(report["manuscript_audit"]["required_markers"].values()) and report["manuscript_audit"]["named_locations"] == ["Title","Abstract","Methods","Results","Limitations","Conclusion"],
        "G17_response_directly_answers_without_forbidden_claims": "do not describe this work as experimental gkp quantum error correction" in text and "every physical measurement field is null" in text and "cannot validate our simulator" in text and not any(phrase.lower() in text for phrase in report["forbidden_response_phrases"]),
        "G18_response_rows_are_lossless_unique_and_state_complete": len(rows) == 24 and len({row["row_id"] for row in rows}) == 24 and {row["response_state"] for row in rows} == RESPONSE_STATES and all(row["claim"] and row["boundary"] and row["source_ids"] for row in rows),
        "G19_every_response_source_id_is_registered": all(set(row["source_ids"]) <= allowed_sources for row in rows),
        "G20_all_artifact_bindings_are_live": (not check_live_sources) or all(_binding_live(binding) for binding in registry.values()),
        "G21_hardware_contribution_is_digital_and_final_gate_stays_preboard": report["final_gate_context"]["verdict"] == "GO_RTL_ONLY" and report["final_gate_context"]["truth_key"] == "multimode=false,rtl=true" and "pre-board" in report["final_gate_context"]["board_claim"] and report["final_gate_context"]["board_measured"] is False,
        "G22_external_validity_gap_and_offline_first_route_are_explicit": report["external_validity"] == {"resolved":False,"current_limit":"no physical FPGA measurement, real GKP syndrome dataset, cavity/transmon control chain or active feedback","phase8_optional":True,"plan_requires_offline_before_closed_loop":True},
        "G23_risk_and_task_audit_are_present": report["risk_audit"] == {"r_n160_present":True,"t7_3_3_audit_present":True},
        "G24_every_ladder_level_has_two_sided_boundary": all(row["allowed"] and row["forbidden"] and row["status"] for row in ladder),
    }
    return gates


def _run_mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    def mutate(path: Sequence[Any], value: Any) -> dict[str, Any]:
        candidate = copy.deepcopy(report)
        target: Any = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return candidate

    cases = [
        ("G01_identity_and_task_handoff_are_exact", lambda: mutate(("task_status","T7.3.3"),"In Progress")),
        ("G02_preemptive_context_and_placeholder_are_honest", lambda: mutate(("response_package","package_readiness"),"ready_to_submit")),
        ("G03_evidence_ladder_is_ordered_and_noncollapsed", lambda: {**copy.deepcopy(report),"evidence_ladder":list(reversed(report["evidence_ladder"]))}),
        ("G04_literature_values_remain_context_only", lambda: mutate(("evidence_ladder",0,"evidence","literature_value_cells"),58)),
        ("G05_official_code_reproduction_is_numerical_not_physical", lambda: mutate(("evidence_ladder",1,"evidence","physical_measurement"),True)),
        ("G06_aqec_project_replay_is_deep_but_not_paper_native", lambda: mutate(("evidence_ladder",2,"evidence","aqec_seed_clusters"),6)),
        ("G07_multimode_simulation_negative_result_stays_visible", lambda: mutate(("evidence_ladder",2,"evidence","multimode_relative_improvement"),0.1)),
        ("G08_software_hil_is_explicitly_mock_not_board", lambda: mutate(("evidence_ladder",3,"evidence","real_board_hil"),True)),
        ("G09_formal_property_and_mutation_counts_are_exact", lambda: mutate(("evidence_ladder",4,"evidence","formal_gates"),16)),
        ("G10_long_rtl_is_million_cycle_full_safety_replay", lambda: mutate(("evidence_ladder",4,"evidence","undefined_actions"),1)),
        ("G11_place_route_is_three_seed_preboard_estimate", lambda: mutate(("evidence_ladder",4,"evidence","measured"),True)),
        ("G12_physical_board_gate_is_blocked_and_all_null", lambda: mutate(("evidence_ladder",5,"evidence","all_null"),False)),
        ("G13_real_gkp_and_control_chain_remain_optional_todo", lambda: mutate(("evidence_ladder",6,"evidence","phase8_statuses","T8.1.1"),"Done")),
        ("G14_all_cross_level_evidence_transfers_are_forbidden", lambda: mutate(("nontransfer_contract","aqec_or_sivak_to_project_physical_evidence"),True)),
        ("G15_title_and_abstract_avoid_experimental_gkp_claim", lambda: mutate(("terminology_contract","experimental_gkp_qec_claim"),True)),
        ("G16_manuscript_discloses_preboard_null_and_physical_lifetime_boundaries", lambda: mutate(("manuscript_audit","required_markers","matched device experiment"),False)),
        ("G17_response_directly_answers_without_forbidden_claims", lambda: mutate(("response_package","english_response"),report["response_package"]["english_response"]+" We experimentally demonstrate.")),
        ("G18_response_rows_are_lossless_unique_and_state_complete", lambda: {**copy.deepcopy(report),"response_rows":list(report["response_rows"][:-1])}),
        ("G19_every_response_source_id_is_registered", lambda: mutate(("response_rows",0,"source_ids"),["unregistered_source"])),
        ("G20_all_artifact_bindings_are_live", lambda: mutate(("artifact_registry","atlas","bytes"),report["artifact_registry"]["atlas"]["bytes"]+1)),
        ("G21_hardware_contribution_is_digital_and_final_gate_stays_preboard", lambda: mutate(("final_gate_context","board_claim"),"physical board qualified")),
        ("G22_external_validity_gap_and_offline_first_route_are_explicit", lambda: mutate(("external_validity","resolved"),True)),
        ("G23_risk_and_task_audit_are_present", lambda: mutate(("risk_audit","r_n160_present"),False)),
        ("G24_every_ladder_level_has_two_sided_boundary", lambda: mutate(("evidence_ladder",6,"forbidden"),"")),
    ]
    results = []
    for target_gate, factory in cases:
        mutated = factory()
        detected = not evaluate_gates(mutated, check_live_sources=target_gate == "G20_all_artifact_bindings_are_live")[target_gate]
        results.append({"mutation_id":f"M{len(results)+1:02d}","target_gate":target_gate,"detected":detected})
    return {"count":len(results),"detected":sum(case["detected"] for case in results),"cases":results}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Reviewer response: experiment relevance without quantum hardware",
        "",
        f"- Task: `{report['task_id']}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Package readiness: `{report['response_package']['package_readiness']}`",
        f"- Gates/mutations: `{report['gate_summary']['passed']}/{report['gate_summary']['total']}` / `{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        "",
        "## Point-by-point response",
        "",
        report["response_package"]["english_response"],
        "",
        "## Evidence ladder",
        "",
        "| Level | Current status | Allowed | Forbidden |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["evidence_ladder"]:
        lines.append(f"| `{row['level']}` | `{row['status']}` | {row['allowed']} | {row['forbidden']} |")
    lines.extend(["", "## Manuscript checklist", ""])
    lines.extend(f"- {item}" for item in report["response_package"]["manuscript_change_checklist"])
    lines.extend(["", "## Missing author input", "", f"- `{report['response_package']['missing_information'][0]}`", "", "## 中文核对", "", "本回答把“实验相关”收紧为“实验启发/面向实验接口”，并逐层区分文献事实、官方代码复现、项目原生仿真、mock 软件 HIL、预板 RTL、真板测量与真实 GKP/量子硬件。AQEC/Sivak 的物理证据不迁移；当前真板字段全部为空，Phase 8 仍为可选 Todo。", ""])
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_outputs(report: dict[str, Any]) -> None:
    rows = _source_rows(report)
    fd, temp_name = tempfile.mkstemp(prefix=f".{DEFAULT_SOURCE_DATA.name}.", suffix=".tmp", dir=DEFAULT_SOURCE_DATA.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, DEFAULT_SOURCE_DATA)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    report["source_data"] = {**_binding(DEFAULT_SOURCE_DATA), "rows":len(rows)}
    _atomic_write_text(DEFAULT_MARKDOWN, _markdown(report))
    report["markdown"] = _binding(DEFAULT_MARKDOWN)
    _atomic_write_text(DEFAULT_REPORT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def verify_report() -> tuple[bool, dict[str, bool]]:
    if not DEFAULT_REPORT.exists():
        return False, {"outputs_exist":False}
    stored = _load(DEFAULT_REPORT)
    fresh = build_report(generated_at_utc=stored.get("generated_at_utc"))
    checks = {
        "outputs_exist": DEFAULT_SOURCE_DATA.exists() and DEFAULT_MARKDOWN.exists(),
        "identity": stored.get("task_id") == TASK_ID and stored.get("schema_version") == SCHEMA_VERSION,
        "verdict": stored.get("verdict") == VERDICT and fresh.get("verdict") == VERDICT,
        "all_gates": all(evaluate_gates(stored, check_live_sources=True).values()),
        "all_mutations": stored["semantic_mutation_audit"]["count"] == stored["semantic_mutation_audit"]["detected"] == len(stored["gates"]),
        "source_data": _source_data_matches(stored),
        "markdown_live": _binding_live(stored["markdown"]),
        "analysis_live": stored.get("analysis_sha256") == _canonical_sha256(_analysis_payload(stored)),
        "fresh_analysis": stored.get("analysis_sha256") == fresh.get("analysis_sha256"),
    }
    return all(checks.values()), checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        ok, checks = verify_report()
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    report = build_report()
    write_outputs(report)
    print(json.dumps({"verdict":report["verdict"],"gates":report["gate_summary"],"mutations":{"detected":report["semantic_mutation_audit"]["detected"],"total":report["semantic_mutation_audit"]["count"]},"source_rows":len(report["response_rows"]),"package_readiness":report["response_package"]["package_readiness"],"analysis_sha256":report["analysis_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
