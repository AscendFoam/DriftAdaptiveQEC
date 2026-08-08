"""T7.2.1 evidence-bounded Introduction and Related Work contract.

The manuscript text is the human-facing output.  This module makes its claim,
citation, task-signature, negative-result, and hardware-boundary obligations
machine-checkable so that later prose edits fail closed instead of silently
upgrading the paper.
"""

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


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.2.1"
SCHEMA_VERSION = "t7.2.1-introduction-related-work-contract-v1"
VERDICT = "PASS_EVIDENCE_BOUNDED_INTRODUCTION_RELATED_WORK"

NOTE_PATH = ROOT / "docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex"
BIB_PATH = ROOT / "docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib"
DEFAULT_REPORT = ROOT / "docs/t7_2_1_introduction_related_work_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_2_1_introduction_related_work_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/introduction_related_work_contract.md"

SOURCE_PATHS = {
    "claim_matrix": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json",
    "method_source_audit": ROOT / "docs/t6_16_1_secondary_method_source_audit.json",
    "metric_ontology": ROOT / "docs/t6_16_2_comparison_ontology.json",
    "literature_matrix": ROOT / "docs/literature_matrix.md",
    "phase6c_integrity": ROOT / "docs/t6_19_3_secondary_evidence_integrity.json",
    "v4_final_gate": ROOT / "docs/t6_9_3_route_a_final_evidence_gate.json",
    "v5_final_gate": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json",
    "gqf_exact_attempt": ROOT / "docs/t6_8_4_gqf_paper_exact_reproduction.json",
    "learned_eligibility": ROOT / "docs/t6_17_3_learned_model_eligibility_replay.json",
    "fpga_normalization": ROOT / "docs/t6_19_2_external_fpga_normalization.json",
    "bibliography": BIB_PATH,
    "implementation": Path(__file__).resolve(),
}

RELATED_SUBSECTIONS = (
    "Analog, history-aware, and structured GKP decoding",
    "Calibration and drift-adaptive decoding",
    "Learned, non-Markovian, and autonomous feedback",
    "Deterministic and FPGA QEC decoders",
    "Position of the present work",
)

LITERATURE_LANES = (
    "single_mode_decoder",
    "surface_gkp_gate_ci_ml",
    "multimode_structured_cpd",
    "direct_nn_rl_nmf_controller",
    "aqec_physical_protocol",
    "fpga_implementation",
)

TASK_SIGNATURE_FIELDS = (
    "code_or_modes",
    "noise_model",
    "syndrome_input",
    "output_action",
    "observability_or_privilege",
    "compute_budget",
    "precision",
    "timing_boundary",
    "evidence_grade",
)

REQUIRED_CITATIONS = {
    "gkp2001", "grimsmo2021", "hastrup2023", "jafarzadeh2025",
    "sivak2023", "lachance2024", "fukui2018", "noh2020", "noh2022",
    "wan2020", "berent2024", "lin2023", "spitz2018", "wagner2021",
    "chen2022", "dgr2023", "sivak2024", "bausch2024", "wang2022",
    "stein2026", "sivak2026", "puviani2025", "lilliput2022",
    "helios2023", "collision2025", "ziad2024", "maurer2025",
    "caune2024", "yang2026", "raveendran2022",
}

PROHIBITED_ASSERTIVE_PATTERNS = (
    "we outperform static gkp",
    "route-a outperforms static gkp",
    "we surpass puviani",
    "route-a surpasses nmf",
    "fastest fpga decoder",
    "state-of-the-art gkp decoder",
    "we are the first adaptive gkp decoder",
    "measured 222.222 ns",
    "measured six-cycle latency",
    "broad tail improvement is established",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return (
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and _sha256(path) == str(binding["sha256"])
    )


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _extract_section(tex: str, title: str) -> str:
    marker = rf"\\section\{{{re.escape(title)}\}}"
    match = re.search(marker, tex)
    if match is None:
        raise ValueError(f"missing section: {title}")
    tail = tex[match.end():]
    next_match = re.search(r"\\section\{", tail)
    return tail[: next_match.start() if next_match else len(tail)].strip()


def _extract_related_subsections(related: str) -> dict[str, str]:
    matches = list(re.finditer(r"\\subsection\{([^}]+)\}", related))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(related)
        result[match.group(1)] = related[match.end():end].strip()
    return result


def _citations(text: str) -> list[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return sorted(keys)


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text))


def _prose_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block.startswith("\\") or "\\begin{" in block:
            continue
        paragraphs.append(block)
    return paragraphs


def _evidence_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "IRW-001", "section": "Introduction", "lane_id": "physical_gkp_context",
            "claim_anchor": "finite-energy GKP and physical feedback motivate an execution-aware decoder boundary",
            "citation_keys": ["gkp2001", "grimsmo2021", "hastrup2023", "jafarzadeh2025", "lachance2024", "sivak2023"],
            "source_ids": ["literature_matrix:A01,A04,A05,A06,A08"], "evidence_grade": "LITERATURE_PRIMARY",
            "boundary": "External theory and experiment motivate the task; they are not project hardware evidence.",
        },
        {
            "row_id": "IRW-002", "section": "Introduction", "lane_id": "single_mode_decoder",
            "claim_anchor": "analog, ML, history-aware, and CPD methods are prior art but have different task signatures",
            "citation_keys": ["fukui2018", "noh2020", "noh2022", "wan2020", "berent2024", "lin2023"],
            "source_ids": ["T6.16.1:method_registry", "T6.16.2:lane_ontology"], "evidence_grade": "LITERATURE_PRIMARY",
            "boundary": "No cross-code CI/ML/MAP/CPD leaderboard is permitted.",
        },
        {
            "row_id": "IRW-003", "section": "Introduction", "lane_id": "drift_calibration",
            "claim_anchor": "syndrome-driven calibration and time-dependent decoder adaptation have established precedents",
            "citation_keys": ["spitz2018", "wagner2021", "chen2022", "dgr2023", "sivak2024", "stein2026"],
            "source_ids": ["literature_matrix:C02-C06,C09"], "evidence_grade": "LITERATURE_PRIMARY",
            "boundary": "The project does not claim the first adaptive or calibration-aware QEC decoder.",
        },
        {
            "row_id": "IRW-004", "section": "Introduction", "lane_id": "contract_system",
            "claim_anchor": "MAP owns LER, typed event/fallback owns safety, and the fast path owns deterministic execution",
            "citation_keys": [], "source_ids": ["T7.1.1:CONTRACT_SYSTEM_INTEGRATION", "T6.9.3:V4_FINAL_GATE"],
            "evidence_grade": "PROJECT_NATIVE_PREBOARD",
            "boundary": "Restricted simulator/pre-board integration only; no measured board or global decoder claim.",
        },
        {
            "row_id": "IRW-005", "section": "Introduction", "lane_id": "single_mode_decoder",
            "claim_anchor": "the positive V4 result is restricted to locked EWMA while static and Window remain counterexamples",
            "citation_keys": [], "source_ids": ["T7.1.1:SMOOTH_LOCKED_EWMA_ADVANTAGE", "T7.1.1:STATIC_GKP_SUPERIORITY"],
            "evidence_grade": "PROJECT_NATIVE_MATCHED",
            "boundary": "No superiority over static or Window MAP and no all-family drift advantage.",
        },
        {
            "row_id": "IRW-006", "section": "Introduction", "lane_id": "v5_negative",
            "claim_anchor": "posterior-predictive V5 stopped before formal/RTL because causal deployable headroom failed",
            "citation_keys": [], "source_ids": ["T6.10.1:CAUSAL_HEADROOM", "T6.15.5:V5_EARLY_STOP"],
            "evidence_grade": "NEGATIVE",
            "boundary": "V5 has no formal, fixed-point, CXXRTL, P&R, or measured-hardware result.",
        },
        {
            "row_id": "IRW-007", "section": RELATED_SUBSECTIONS[0], "lane_id": "surface_gkp_gate_ci_ml",
            "claim_anchor": "surface-GKP CI/ML and multimode CPD are method-specific lanes",
            "citation_keys": ["noh2020", "noh2022", "raveendran2022", "berent2024", "lin2023"],
            "source_ids": ["T6.17.2:NOH_CNOT_REPRODUCTION", "T6.18.2:OFFICIAL_CPD"],
            "evidence_grade": "LITERATURE_PLUS_OFFICIAL_REPRODUCTION",
            "boundary": "Gate failure, threshold, and multimode LER cannot be subtracted from single-mode Route-A LER.",
        },
        {
            "row_id": "IRW-008", "section": RELATED_SUBSECTIONS[1], "lane_id": "drift_calibration",
            "claim_anchor": "the contract contribution is update authority and cost accounting, not a new estimator class",
            "citation_keys": ["spitz2018", "wagner2021", "chen2022", "dgr2023", "sivak2024", "stein2026"],
            "source_ids": ["T6.5.2:UNIFIED_EXECUTION_CONTRACT"], "evidence_grade": "SYNTHESIS_INFERENCE",
            "boundary": "Novelty is joint systems/evidence integration, not absolute algorithmic first use.",
        },
        {
            "row_id": "IRW-009", "section": RELATED_SUBSECTIONS[2], "lane_id": "direct_nn_rl_nmf_controller",
            "claim_anchor": "Direct NN, model-free RL, model-based NMF, and AQEC have different actions and privileges",
            "citation_keys": ["bausch2024", "wang2022", "sivak2026", "puviani2025", "lachance2024"],
            "source_ids": ["T6.16.1:controller_registry"], "evidence_grade": "LITERATURE_PRIMARY",
            "boundary": "Controller gain, decoder LER, and autonomous lifetime are never merged.",
        },
        {
            "row_id": "IRW-010", "section": RELATED_SUBSECTIONS[2], "lane_id": "direct_nn_rl_nmf_controller",
            "claim_anchor": "NMF exact reproduction is ineligible and learned same-task eligibility is zero",
            "citation_keys": ["puviani2025"],
            "source_ids": ["T6.8.4:GQF_EXACT_NO_GO", "T6.17.3:LEARNED_ELIGIBLE_ZERO", "T7.1.1:P6C-NMF-GQF"],
            "evidence_grade": "NEGATIVE",
            "boundary": "No claim of reproducing or surpassing Puviani NMF is allowed.",
        },
        {
            "row_id": "IRW-011", "section": RELATED_SUBSECTIONS[3], "lane_id": "fpga_implementation",
            "claim_anchor": "FPGA comparisons require matched code, precision, timing boundary, and evidence grade",
            "citation_keys": ["lilliput2022", "helios2023", "collision2025", "ziad2024", "maurer2025", "yang2026", "caune2024"],
            "source_ids": ["T6.19.2:EXTERNAL_FPGA_NORMALIZATION"], "evidence_grade": "LITERATURE_PRIMARY",
            "boundary": "Core, per-round, source-to-action, and closed-loop latency are not interchangeable.",
        },
        {
            "row_id": "IRW-012", "section": RELATED_SUBSECTIONS[3], "lane_id": "fpga_implementation",
            "claim_anchor": "zero exact same-task FPGA comparators prevents, rather than supports, a speed ranking",
            "citation_keys": [],
            "source_ids": ["T6.19.2:SAME_TASK_COMPARATOR_ZERO", "T7.1.1:FPGA_SPEED_ADVANTAGE"],
            "evidence_grade": "NEGATIVE",
            "boundary": "Six cycles/II=1 and post-route values are estimates, not measured board superiority.",
        },
        {
            "row_id": "IRW-013", "section": RELATED_SUBSECTIONS[4], "lane_id": "positioning",
            "claim_anchor": "the innovation unit is the evidence-gated contract rather than a CNN-centric decoder",
            "citation_keys": [],
            "source_ids": ["T7.1.1:MANUSCRIPT_DECISION", "T6.19.3:AUX_INTEGRITY"],
            "evidence_grade": "PROJECT_SYNTHESIS",
            "boundary": "Auxiliary positive results remain task-local and cannot rescue V4/V5 promotion.",
        },
        {
            "row_id": "IRW-014", "section": RELATED_SUBSECTIONS[4], "lane_id": "positioning",
            "claim_anchor": "negative evidence and null hardware fields remain visible in the paper narrative",
            "citation_keys": [],
            "source_ids": ["T7.1.1:MANDATORY_NEGATIVES", "T7.1.3:MAIN_RESULTS", "T7.1.4:FAILURE_LEDGER"],
            "evidence_grade": "PROJECT_SYNTHESIS",
            "boundary": "The manuscript is restricted pre-board, not a cross-protocol or experimental GKP ranking.",
        },
    ]


def _section_order(tex: str) -> list[str]:
    return re.findall(r"\\section\{([^}]+)\}", tex)


def _build_manuscript_snapshot() -> dict[str, Any]:
    tex = NOTE_PATH.read_text(encoding="utf-8")
    intro = _extract_section(tex, "Introduction")
    related = _extract_section(tex, "Related Work")
    subsections = _extract_related_subsections(related)
    cited = _citations(intro + "\n" + related)
    bib_keys = sorted(_bib_keys(BIB_PATH.read_text(encoding="utf-8")))
    return {
        "note_path": _relative(NOTE_PATH),
        "section_order": _section_order(tex),
        "introduction": intro,
        "introduction_sha256": hashlib.sha256(intro.encode("utf-8")).hexdigest(),
        "introduction_paragraphs": len(_prose_paragraphs(intro)),
        "related_work": related,
        "related_work_sha256": hashlib.sha256(related.encode("utf-8")).hexdigest(),
        "related_subsections": subsections,
        "related_subsection_citations": {name: _citations(text) for name, text in subsections.items()},
        "citation_keys": cited,
        "bibliography_keys": bib_keys,
    }


def _gates(payload: Mapping[str, Any]) -> dict[str, bool]:
    manuscript = payload["manuscript"]
    intro = _normalize(str(manuscript["introduction"]))
    related = _normalize(str(manuscript["related_work"]))
    combined = intro + " " + related
    subsections = manuscript["related_subsections"]
    rows = payload["claim_evidence_rows"]
    lane_ids = set(payload["comparison_contract"]["literature_lanes"])
    order = list(manuscript["section_order"])
    try:
        order_ok = order.index("Introduction") < order.index("Scope, argument, and evidence cutoff") < order.index("Related Work") < order.index("Contract-centric dual-loop method")
    except ValueError:
        order_ok = False
    subsection_citations = manuscript["related_subsection_citations"]
    required_role_phrases = (
        "a locked map expert owns the logical-error decision",
        "typed event and fallback logic owns tail safety",
        "the six-cycle, initiation-interval-one fast path executes only validated integer state",
    )
    gates = {
        "source_bindings_live": all(_binding_live(binding) for binding in payload["source_bindings"].values()),
        "section_order": order_ok,
        "introduction_structure": 6 <= int(manuscript["introduction_paragraphs"]) <= 7 and len(intro.split()) >= 450,
        "related_work_structure": tuple(subsections) == RELATED_SUBSECTIONS and all(len(_normalize(text).split()) >= 70 for text in subsections.values()),
        "citation_keys_resolve": set(manuscript["citation_keys"]).issubset(set(manuscript["bibliography_keys"])),
        "citation_coverage": REQUIRED_CITATIONS.issubset(set(manuscript["citation_keys"])) and all(len(subsection_citations.get(name, ())) >= (0 if name == RELATED_SUBSECTIONS[4] else 2) for name in RELATED_SUBSECTIONS),
        "claim_rows_traceable": all(row.get("boundary") and row.get("evidence_grade") and (row.get("citation_keys") or row.get("source_ids")) for row in rows),
        "literature_lanes_complete": lane_ids == set(LITERATURE_LANES),
        "task_signature_contract_explicit": payload["comparison_contract"].get("task_signature_required") is True and tuple(payload["comparison_contract"].get("task_signature_fields", ())) == TASK_SIGNATURE_FIELDS and "task-signature lanes" in related,
        "static_window_negative_visible": "static joint map has a lower average error rate" in intro and "window map remains a stronger counterexample" in intro,
        "v5_early_stop_visible": "stopped before formal or rtl work" in intro and "causal deployable headroom gates failed" in intro,
        "nmf_boundary_visible": "did not support a paper-exact matched reproduction" in related and "zero learned/controller entries eligible" in related and "surpasses nmf" in related,
        "learned_eligibility_visible": "zero learned/controller entries eligible" in related and "no matched learned checkpoint was eligible" in related,
        "fpga_boundary_visible": "a count of zero same-task comparators prevents a fair speed ranking" in related and "it does not imply that the project is faster" in related and "remain unmeasured" in intro,
        "global_ranking_prohibited": payload["comparison_contract"].get("global_ranking_allowed") is False and "prevents heterogeneous evidence from becoming a global ranking" in intro and "global comparison across ci, ml, direct nn, rl, aqec, cpd, and fpga systems" in related,
        "role_terminology_consistent": all(phrase in intro for phrase in required_role_phrases),
        "prohibited_assertions_absent": not any(pattern in combined for pattern in PROHIBITED_ASSERTIVE_PATTERNS),
        "source_data_rows_complete": len(rows) == 14 and len({row["row_id"] for row in rows}) == 14,
    }
    return gates


def _mutate(payload: Mapping[str, Any], gate: str) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    manuscript = value["manuscript"]
    if gate == "source_bindings_live":
        value["source_bindings"]["claim_matrix"]["sha256"] = "0" * 64
    elif gate == "section_order":
        manuscript["section_order"] = list(reversed(manuscript["section_order"]))
    elif gate == "introduction_structure":
        manuscript["introduction_paragraphs"] = 1
    elif gate == "related_work_structure":
        manuscript["related_subsections"].pop(RELATED_SUBSECTIONS[0])
    elif gate == "citation_keys_resolve":
        manuscript["citation_keys"].append("fabricated_missing_key")
    elif gate == "citation_coverage":
        manuscript["citation_keys"] = [key for key in manuscript["citation_keys"] if key != "puviani2025"]
    elif gate == "claim_rows_traceable":
        value["claim_evidence_rows"][0]["citation_keys"] = []
        value["claim_evidence_rows"][0]["source_ids"] = []
    elif gate == "literature_lanes_complete":
        value["comparison_contract"]["literature_lanes"] = list(LITERATURE_LANES[:-1])
    elif gate == "task_signature_contract_explicit":
        value["comparison_contract"]["task_signature_required"] = False
    elif gate == "static_window_negative_visible":
        manuscript["introduction"] = manuscript["introduction"].replace("static joint MAP has a lower average error rate", "static joint MAP is competitive")
    elif gate == "v5_early_stop_visible":
        manuscript["introduction"] = manuscript["introduction"].replace("stopped before formal or RTL work", "continued through RTL")
    elif gate == "nmf_boundary_visible":
        manuscript["related_work"] = re.sub(
            r"did\s+not\s+support\s+a\s+paper-exact\s+matched\s+reproduction",
            "supported a paper-exact matched reproduction",
            manuscript["related_work"],
            flags=re.IGNORECASE,
        )
    elif gate == "learned_eligibility_visible":
        manuscript["related_work"] = manuscript["related_work"].replace("zero learned/controller entries eligible", "several learned/controller entries eligible")
    elif gate == "fpga_boundary_visible":
        manuscript["related_work"] = re.sub(
            r"A\s+count\s+of\s+zero\s+same-task\s+comparators\s+prevents\s+a\s+fair\s+speed\s+ranking",
            "A count of zero same-task comparators supports a speed ranking",
            manuscript["related_work"],
            flags=re.IGNORECASE,
        )
    elif gate == "global_ranking_prohibited":
        value["comparison_contract"]["global_ranking_allowed"] = True
    elif gate == "role_terminology_consistent":
        manuscript["introduction"] = re.sub(
            r"A\s+locked\s+MAP\s+expert\s+owns\s+the\s+logical-error\s+decision",
            "A CNN owns the logical-error decision",
            manuscript["introduction"],
            flags=re.IGNORECASE,
        )
    elif gate == "prohibited_assertions_absent":
        manuscript["related_work"] += "\n\nWe are the first adaptive GKP decoder."
    elif gate == "source_data_rows_complete":
        value["claim_evidence_rows"].pop()
    return value


def _semantic_mutation_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    cases = []
    for gate in _gates(payload):
        mutated = _mutate(payload, gate)
        rejected = _gates(mutated).get(gate) is False
        cases.append({"mutation_id": f"MUT-{len(cases)+1:02d}", "target_gate": gate, "rejected": rejected})
    return {
        "count": len(cases),
        "detected": sum(bool(case["rejected"]) for case in cases),
        "cases": cases,
    }


def build_report(*, generated_at_utc: str | None = None) -> dict[str, Any]:
    rows = _evidence_rows()
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "paper_argument": "For repeated approximate-GKP correction, decoder adaptation is an evidence-gated execution contract: MAP owns LER, typed event/fallback logic owns safety, and a versioned six-cycle fast path owns deterministic execution.",
        "writing_axes": {
            "paper_type": "algorithmic systems paper",
            "sections": ["Introduction", "Related Work"],
            "language": "English drafted from Chinese evidence ledger",
            "journal_style": "generic high-impact / Nature-leaning argument discipline",
            "introduction_variant": "general-to-specific plus open-with-challenge",
            "related_work_variant": "mechanism-grouped with explicit task signatures",
        },
        "comparison_contract": {
            "literature_lanes": list(LITERATURE_LANES),
            "task_signature_required": True,
            "task_signature_fields": list(TASK_SIGNATURE_FIELDS),
            "global_ranking_allowed": False,
            "literature_values_as_project_results": False,
            "same_task_zero_implies_superiority": False,
        },
        "manuscript": _build_manuscript_snapshot(),
        "claim_evidence_rows": rows,
        "source_bindings": {key: _binding(path) for key, path in SOURCE_PATHS.items()},
        "boundary_summary": {
            "current_manuscript_state": "RESTRICTED_PREBOARD_CONTRACT_PAPER",
            "v5_state": "NO_GO_V5_EARLY_HEADROOM_STOP",
            "board_state": "BLOCKED_ALL_MEASURED_FIELDS_NULL",
            "learned_same_task_eligible": 0,
            "external_fpga_same_task_comparators": 0,
            "nmf_exact_reproduction": "INELIGIBLE_SOURCE_INCOMPLETE",
            "tail_claim": "LOCKED_EWMA_NONINFERIORITY_NOT_BROAD_IMPROVEMENT",
        },
    }
    payload["gates"] = _gates(payload)
    payload["gate_summary"] = {"passed": sum(payload["gates"].values()), "total": len(payload["gates"])}
    payload["semantic_mutation_audit"] = _semantic_mutation_audit(payload)
    payload["verdict"] = VERDICT if all(payload["gates"].values()) and payload["semantic_mutation_audit"]["detected"] == len(payload["gates"]) else "FAIL_T7_2_1_CONTRACT"
    hash_payload = copy.deepcopy(payload)
    hash_payload.pop("analysis_sha256", None)
    payload["analysis_sha256"] = _canonical_sha256(hash_payload)
    return payload


def _write_source_data(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = ("row_id", "section", "lane_id", "claim_anchor", "evidence_grade", "boundary", "citation_keys_json", "source_ids_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{field: row[field] for field in fields[:6]},
                "citation_keys_json": json.dumps(row["citation_keys"], ensure_ascii=False, separators=(",", ":")),
                "source_ids_json": json.dumps(row["source_ids"], ensure_ascii=False, separators=(",", ":")),
            })
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    manuscript = report["manuscript"]
    lines = [
        "# T7.2.1 Introduction / Related Work 证据合同",
        "",
        f"- 状态：`{report['verdict']}`",
        f"- 主论点：{report['paper_argument']}",
        f"- 正文源：`{manuscript['note_path']}`",
        f"- Introduction：{manuscript['introduction_paragraphs']} 个 prose 段，section SHA-256 `{manuscript['introduction_sha256']}`",
        f"- Related Work：{len(manuscript['related_subsections'])} 个 mechanism group，section SHA-256 `{manuscript['related_work_sha256']}`",
        f"- 引用：{len(manuscript['citation_keys'])} 个已解析 key；机器合同 {report['gate_summary']['passed']}/{report['gate_summary']['total']} gates，语义篡改 {report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']} 检出。",
        "",
        "## 写作结构与边界",
        "",
        "Introduction 采用 general-to-specific + open-with-challenge：从 finite-energy GKP 与真实反馈约束，依次收敛到 analog/history、drift/calibration、timing-boundary，再提出 execution-contract 问题和当前 restricted 结论。Related Work 按机制分组，不按截图中的类别做总榜。",
        "",
        "必须保留的负证据：static joint MAP 平均 LER 更低、Window MAP 是强反例、tail 只通过 locked-EWMA non-inferiority、V5 在 headroom 门提前停止、NMF exact reproduction 不合格、learned same-task eligibility=0、external FPGA same-task comparator=0、实板字段仍为 null。",
        "",
        "## Claim / citation / evidence 行",
        "",
        "| Row | Section | Lane | Evidence | Citation / project source | Boundary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["claim_evidence_rows"]:
        trace = ", ".join(row["citation_keys"] or row["source_ids"])
        lines.append(f"| {row['row_id']} | {row['section']} | `{row['lane_id']}` | `{row['evidence_grade']}` | {trace} | {row['boundary']} |")
    lines.extend([
        "",
        "## 反简化检查",
        "",
        "- 不是只检查章节标题：合同同时验证段落结构、五个机制组、30 个必需 citation key、14 条 claim-evidence 行和六条 comparison lane。",
        "- 不是关键词 smoke：18 个 gate 各有定向语义篡改，删除 static/Window 负结果、把 V5 改成继续实现、升级 NMF/learned/FPGA 或加入绝对首次主张都会被拒绝。",
        "- 引用从独立 BibTeX 文件解析；任何正文 citation key 缺失都会 fail closed。",
        "- 文献值、official-code reproduction、project-native simulation、P&R estimate 与 board measurement 不得互换。",
        "",
        "## 论文 claim 影响",
        "",
        "该任务把旧 CNN-centric 叙事收敛为 contract-centric、regime-aware 的安全双回路，但没有升级性能 verdict。当前可写的是 restricted simulator/pre-board integration；不可写的是 static/Window superiority、broad tail gain、Puviani NMF surpass、真实 break-even、fastest FPGA 或 measured board result。",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any]) -> None:
    _write_source_data(report["claim_evidence_rows"], DEFAULT_SOURCE_DATA)
    _atomic_json(report, DEFAULT_REPORT)
    _atomic_text(_markdown(report), DEFAULT_MARKDOWN)


def verify_report(path: Path = DEFAULT_REPORT) -> dict[str, bool]:
    stored = json.loads(path.read_text(encoding="utf-8"))
    live = build_report(generated_at_utc=stored.get("generated_at_utc"))
    keys = (
        "paper_argument", "writing_axes", "comparison_contract", "manuscript",
        "claim_evidence_rows", "source_bindings", "boundary_summary", "gates",
        "gate_summary", "semantic_mutation_audit", "verdict", "analysis_sha256",
    )
    checks = {f"live_{key}": stored.get(key) == live.get(key) for key in keys}
    checks["all_stored_gates_pass"] = all(stored.get("gates", {}).values())
    checks["mutation_audit_complete"] = stored.get("semantic_mutation_audit", {}).get("detected") == len(stored.get("gates", {}))
    checks["verdict"] = stored.get("verdict") == VERDICT
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify the committed report against live sources")
    args = parser.parse_args(argv)
    if args.verify:
        checks = verify_report()
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if all(checks.values()) else 1
    report = build_report()
    write_outputs(report)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "mutations": report["semantic_mutation_audit"]}, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
