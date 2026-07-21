"""T7.2.6 machine contract for the Phase-6D dual-lane manuscript delta.

The canonical manuscript must consume the final T6.26.4/T7.1.5 truth table
without reviving the historical Phase-6C narrative.  The multimode lane keeps
the strongest-baseline no-go and unopened formal fields visible; the exact
single-mode RTL lane keeps pre-board timing/safety evidence separate; learning
remains a dropped approximation.  This module binds prose, figures, citations,
parents and historical snapshots and applies independent semantic mutations.
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
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.2.6"
SCHEMA_VERSION = "t7.2.6-phase6d-manuscript-delta-v1"
VERDICT = "PASS_PHASE6D_DUAL_LANE_MANUSCRIPT_DELTA_RTL_ONLY"

CONFIG_PATH = ROOT / "configs/phase6d/t7_2_6_manuscript_delta.json"
DEFAULT_REPORT = ROOT / "docs/t7_2_6_phase6d_manuscript_delta.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_2_6_phase6d_manuscript_delta_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/phase6d_manuscript_delta.md"

PARENT_PATHS = {
    "figure_delta": ROOT / "docs/t7_1_5_phase6d_claim_figure_delta.json",
    "final_gate": ROOT / "docs/t6_26_4_final_dual_lane_gate.json",
    "multimode_headroom": ROOT / "docs/t6_20_4_multimode_causal_headroom.json",
    "rtl_formal": ROOT / "docs/t6_25_2_converged_rtl_formal.json",
    "rtl_long": ROOT / "docs/t6_25_3_converged_long_rtl.json",
    "rtl_hardware": ROOT / "docs/t6_25_4_converged_hardware.json",
}


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    text = re.sub(r"(?<!\\)%[^\n]*", " ", text)
    return " ".join(text.lower().split())


def _extract_environment(tex: str, name: str) -> str:
    match = re.search(rf"\\begin\{{{re.escape(name)}\}}(.*?)\\end\{{{re.escape(name)}\}}", tex, re.S)
    if not match:
        raise ValueError(f"missing environment: {name}")
    return match.group(1)


def _extract_section(tex: str, title: str) -> str:
    match = re.search(rf"\\section\{{{re.escape(title)}\}}", tex)
    if not match:
        raise ValueError(f"missing section: {title}")
    tail = tex[match.end():]
    next_match = re.search(r"\\section\{", tail)
    return tail[: next_match.start() if next_match else len(tail)]


def _contains_all(text: str, tokens: list[str]) -> bool:
    normalized = _normalize(text)
    return all(_normalize(token) in normalized for token in tokens)


def _citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", tex):
        keys.update(item.strip() for item in group.split(",") if item.strip())
    return keys


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", text, re.M))


def _board_status(board: str, task_id: str) -> str | None:
    match = re.search(rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|", board, re.M)
    return match.group(1).strip() if match else None


def _artifact_registry(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {
        "config": CONFIG_PATH,
        "implementation": Path(__file__).resolve(),
        "canonical_manuscript": ROOT / str(config["canonical_manuscript"]),
        "historical_manuscript": ROOT / str(config["historical_manuscript"]),
        "compiled_pdf": ROOT / str(config["compiled_pdf"]),
        "visual_qa": ROOT / str(config["visual_qa"]),
        "bibliography": ROOT / "docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib",
        "phase6d_bibliography": ROOT / "docs/paper_notes/Phase6D_Dual_Lane_GKP_refs.bib",
        "baseline_registry": ROOT / "docs/multimode_strong_baseline_registry.md",
        "task_board": ROOT / "docs/new_task_board.md",
    }
    paths.update(PARENT_PATHS)
    for index, figure_path in enumerate(config["required_figures"].values(), start=1):
        paths[f"figure_{index}"] = ROOT / str(figure_path)
    for index, snapshot in enumerate(config["historical_snapshots"], start=1):
        paths[f"historical_snapshot_{index:02d}"] = ROOT / str(snapshot)
    return {key: _binding(path) for key, path in paths.items()}


def _visual_qa_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    qa = _load_json(ROOT / str(config["visual_qa"]))
    pdf = ROOT / str(config["compiled_pdf"])
    tex = ROOT / str(config["canonical_manuscript"])
    log = pdf.with_suffix(".log")
    diagnostics = qa["log_diagnostics"]
    manual = qa["manual_page_review"]
    rasters = qa["page_raster_checks"]
    return {
        "payload": qa,
        "pdf_hash_live": qa["pdf_sha256"] == _sha256(pdf),
        "tex_hash_live": qa["tex_sha256"] == _sha256(tex),
        "log_hash_live": qa["log_sha256"] == _sha256(log),
        "page_count_exact": qa["page_count"] == len(rasters) == 12,
        "zero_log_diagnostics": all(value == 0 for value in diagnostics.values()),
        "manual_pass": (
            manual["verdict"] == "PASS"
            and manual["pages_reviewed"] == manual["pages_total"] == 12
            and all(manual[key] == 0 for key in (
                "clipping", "overlap", "garbled_glyphs", "orphan_headings", "unexpected_blank_pages"
            ))
            and all(manual[key] is True for key in (
                "figure5_legible", "figure6_legible", "longtable_headers_repeat", "reference_pages_legible"
            ))
        ),
        "raster_bounds": (
            [item["page"] for item in rasters] == list(range(1, 13))
            and all(0.03 <= item["nonwhite_fraction"] <= 0.20 for item in rasters)
            and all(item["edge_ink_fraction"] == 0.0 for item in rasters)
        ),
        "text_scan_pass": (
            qa["text_extraction"]["placeholder_patterns"] == []
            and qa["text_extraction"]["exact_mld_and_kwmw_references_present"] is True
        ),
    }


def _manuscript_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / str(config["canonical_manuscript"])
    historical = ROOT / str(config["historical_manuscript"])
    tex = path.read_text(encoding="utf-8")
    normalized = _normalize(tex)
    headings = re.findall(r"\\section\{([^}]+)\}", tex)
    abstract = _extract_environment(tex, "abstract")
    sections = {title: _extract_section(tex, title) for title in config["section_order"]}
    supplement = "\n".join(value for key, value in sections.items() if key.startswith("Supplementary delta:"))
    contract_text = {
        "Abstract": abstract,
        "Introduction": sections["Introduction"],
        "Methods": sections["Methods"],
        "Results": sections["Results"],
        "Discussion": sections["Discussion"],
        "Limitations": sections["Limitations"],
        "Conclusion": sections["Conclusion"],
        "Supplement": supplement,
    }
    section_checks = {
        name: _contains_all(contract_text[name], list(tokens))
        for name, tokens in config["section_contracts"].items()
    }
    citations = sorted(_citation_keys(tex))
    bib_paths = (
        ROOT / "docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib",
        ROOT / "docs/paper_notes/Phase6D_Dual_Lane_GKP_refs.bib",
    )
    available_bib_keys = set().union(*(_bib_keys(item.read_text(encoding="utf-8")) for item in bib_paths))
    required_citations = set(config["required_citation_keys"])
    figures = {
        label: {
            "path": str(figure_path),
            "exists": (ROOT / str(figure_path)).is_file(),
            "included": Path(str(figure_path)).name in tex,
            "label_present": rf"\label{{{label}}}" in tex,
        }
        for label, figure_path in config["required_figures"].items()
    }
    forbidden_hits = [pattern for pattern in config["forbidden_assertive_patterns"] if _normalize(pattern) in normalized]
    return {
        "path": _relative(path),
        "historical_path": _relative(historical),
        "distinct_from_historical": _sha256(path) != _sha256(historical),
        "section_headings": headings,
        "expected_section_order": list(config["section_order"]),
        "section_order_exact": headings == list(config["section_order"]),
        "section_checks": section_checks,
        "citation_keys": citations,
        "required_citation_keys": sorted(required_citations),
        "required_citations_present": required_citations <= set(citations),
        "all_citations_resolved": set(citations) <= available_bib_keys,
        "figures": figures,
        "forbidden_assertive_hits": forbidden_hits,
        "canonical_terms_present": {
            key: _normalize(str(term)) in normalized for key, term in config["canonical_terms"].items()
        },
    }


def _parent_snapshot(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    delta = parents["figure_delta"]
    final_gate = parents["final_gate"]
    headroom = parents["multimode_headroom"]
    formal = parents["rtl_formal"]
    long_run = parents["rtl_long"]
    hardware = parents["rtl_hardware"]
    claims = copy.deepcopy(delta["claims"])
    claim_by_id = {claim["claim_id"]: claim for claim in claims}
    return {
        "verdicts": {key: value["verdict"] for key, value in parents.items()},
        "final_claims": claims,
        "claim_ids": sorted(claim_by_id),
        "figure_delta_analysis_sha256": delta["analysis_sha256"],
        "final_gate_analysis_sha256": final_gate["analysis_sha256"],
        "multimode": {
            "strongest_baseline": claim_by_id["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_evidence"]["strongest_baseline"],
            "baseline_p_L": claim_by_id["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_evidence"]["baseline_p_L"],
            "proposed_p_L": claim_by_id["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_evidence"]["proposed_p_L"],
            "relative_improvement_point": claim_by_id["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_evidence"]["relative_improvement_point"],
            "relative_improvement_lcb": claim_by_id["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_evidence"]["relative_improvement_lcb"],
            "formal_or_pilot_accessed": claim_by_id["MM_V1_CAUSAL_HEADROOM_NO_GO"]["current_evidence"]["formal_or_pilot_accessed"],
            "headroom_gate": copy.deepcopy(headroom["headroom_gate"]),
        },
        "rtl": {
            "latency_cycles": claim_by_id["RTL_DETERMINISTIC_SIX_CYCLE_II1"]["current_evidence"]["latency_cycles"],
            "ii_cycles": claim_by_id["RTL_DETERMINISTIC_SIX_CYCLE_II1"]["current_evidence"]["initiation_interval_cycles"],
            "cycles": long_run["aggregate_python"]["cycles"],
            "mismatches": claim_by_id["RTL_DETERMINISTIC_SIX_CYCLE_II1"]["current_evidence"]["mismatches"],
            "ii1_input_pairs": long_run["aggregate_python"]["ii1_input_pairs"],
            "ii1_output_pairs": long_run["aggregate_python"]["ii1_output_pairs"],
            "formal_gate_summary": formal["gate_summary"],
            "formal_mutation_summary": formal["mutation_summary"],
            "long_gate_summary": long_run["gate_summary"],
            "minimum_fmax_mhz": hardware["fmax_mhz"]["minimum"],
            "median_fmax_mhz": hardware["fmax_mhz"]["median"],
            "resource_summary": hardware["resource_summary"],
            "board_measured": hardware["evidence_boundary"]["board_measured"],
            "multimode_decoder_in_rtl": hardware["evidence_boundary"]["multimode_decoder_in_rtl"],
            "measured_fields": hardware["measured_fields"],
        },
        "learning": {
            "claim": claim_by_id["LEARNING_APPROXIMATION_DROPPED"],
            "primary": delta["bundle_boundary"]["learning_primary"],
        },
        "nontransfer": {
            "claim": claim_by_id["DUAL_LANE_NONTRANSFERABILITY"],
            "global_weighted_score": delta["bundle_boundary"]["global_weighted_score"],
            "cross_lane_visual_edges": delta["bundle_boundary"]["cross_lane_visual_edges"],
            "one_lane_cannot_satisfy_another": final_gate["decision_policy"] == "INDEPENDENT_BOOLEAN_LANES_NO_WEIGHTED_SCORE_NO_GATE_SUBSTITUTION",
        },
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def append(kind: str, item_id: str, payload: Any) -> None:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        rows.append({
            "row_id": f"{kind}:{item_id}",
            "kind": kind,
            "item_id": item_id,
            "payload_json": canonical,
            "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        })

    for claim in report["parent_state"]["final_claims"]:
        append("claim", claim["claim_id"], claim)
    for section, passed in report["manuscript"]["section_checks"].items():
        append("section_contract", section, {"passed": passed})
    for term, present in report["manuscript"]["canonical_terms_present"].items():
        append("terminology", term, {"present": present})
    for label, figure in report["manuscript"]["figures"].items():
        append("figure", label, figure)
    for key, binding in report["artifact_registry"].items():
        append("artifact", key, binding)
    append("visual_qa", "summary", report["visual_qa"])
    return rows


def _rows_lossless(rows: list[Mapping[str, str]]) -> bool:
    if len({row["row_id"] for row in rows}) != len(rows):
        return False
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            return False
        if _canonical_sha256(payload) != row["payload_sha256"]:
            return False
    return True


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    cfg = report["config_snapshot"]
    manuscript = report["manuscript"]
    parent = report["parent_state"]
    expected = cfg["expected_parent_verdicts"]
    exact_claim_ids = sorted(cfg["required_claim_ids"])
    board_values = parent["rtl"]["measured_fields"]
    gates = {
        "G01_config_identity": report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION,
        "G02_parent_verdicts_exact": parent["verdicts"] == expected,
        "G03_all_artifacts_live": all(_binding_live(binding) for binding in report["artifact_registry"].values()),
        "G04_canonical_is_distinct_delta": manuscript["distinct_from_historical"],
        "G05_section_order_exact": manuscript["section_order_exact"],
        "G06_abstract_contract": manuscript["section_checks"]["Abstract"],
        "G07_introduction_contract": manuscript["section_checks"]["Introduction"],
        "G08_methods_contract": manuscript["section_checks"]["Methods"],
        "G09_results_contract": manuscript["section_checks"]["Results"],
        "G10_discussion_contract": manuscript["section_checks"]["Discussion"],
        "G11_limitations_contract": manuscript["section_checks"]["Limitations"],
        "G12_conclusion_contract": manuscript["section_checks"]["Conclusion"],
        "G13_supplement_contract": manuscript["section_checks"]["Supplement"],
        "G14_final_claim_ids_exact": parent["claim_ids"] == exact_claim_ids,
        "G15_all_claims_placed": all(claim["paper_placements"] for claim in parent["final_claims"]),
        "G16_required_citations_present": manuscript["required_citations_present"],
        "G17_all_citations_resolved": manuscript["all_citations_resolved"],
        "G18_figures_exact": all(all((value["exists"], value["included"], value["label_present"])) for value in manuscript["figures"].values()),
        "G19_forbidden_assertions_absent": manuscript["forbidden_assertive_hits"] == [],
        "G20_parent_numbers_exact": bool(report["exact_parent_number_checks"] and all(report["exact_parent_number_checks"].values())),
        "G21_nontransferability_explicit": (
            parent["nontransfer"]["global_weighted_score"] is None
            and parent["nontransfer"]["cross_lane_visual_edges"] == 0
            and parent["nontransfer"]["one_lane_cannot_satisfy_another"] is True
        ),
        "G22_board_values_are_null": parent["rtl"]["board_measured"] is False and all(value is None for value in board_values.values()),
        "G23_learning_dropped_only": parent["learning"]["primary"] is False and parent["learning"]["claim"]["final_disposition"] == "DROPPED_ABLATION_ONLY",
        "G24_historical_snapshots_live": all(_binding_live(binding) for key, binding in report["artifact_registry"].items() if key.startswith("historical_snapshot_")),
        "G25_source_data_lossless": _rows_lossless(report["source_data"]),
        "G26_board_task_state_valid": report["board_state"] in {"In Progress", "Done"},
        "G27_compiled_pdf_visual_qa": all(
            value for key, value in report["visual_qa"].items() if key != "payload"
        ),
    }
    return gates


def _exact_parent_number_checks(parent: Mapping[str, Any]) -> dict[str, bool]:
    mm = parent["multimode"]
    rtl = parent["rtl"]
    return {
        "mm_baseline": mm["strongest_baseline"] == "static_mixture_exact_mld",
        "mm_equal_ler": mm["baseline_p_L"] == mm["proposed_p_L"] == 0.11197916666666667,
        "mm_zero_headroom": mm["relative_improvement_point"] == mm["relative_improvement_lcb"] == 0.0,
        "mm_downstream_unopened": mm["formal_or_pilot_accessed"] is False,
        "rtl_latency": rtl["latency_cycles"] == 6 and rtl["ii_cycles"] == 1,
        "rtl_long": rtl["cycles"] == 1_000_000 and rtl["mismatches"] == 0,
        "rtl_ii_pairs": rtl["ii1_input_pairs"] == rtl["ii1_output_pairs"] == 998_435,
        "rtl_formal_gates": rtl["formal_gate_summary"] == {"passed": 17, "total": 17},
        "rtl_formal_mutations": rtl["formal_mutation_summary"]["killed"] == rtl["formal_mutation_summary"]["total"] == 21,
        "rtl_long_gates": rtl["long_gate_summary"] == {"passed": 19, "total": 19},
        "rtl_fmax": abs(rtl["minimum_fmax_mhz"] - 36.79446792602539) < 1e-12,
        "rtl_not_multimode": rtl["multimode_decoder_in_rtl"] is False,
    }


def _semantic_mutations(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "G01_config_identity": lambda x: x.update(task_id="T7.2.5"),
        "G02_parent_verdicts_exact": lambda x: x["parent_state"]["verdicts"].update(final_gate="GO_BOTH"),
        "G03_all_artifacts_live": lambda x: x["artifact_registry"]["canonical_manuscript"].update(sha256="0" * 64),
        "G04_canonical_is_distinct_delta": lambda x: x["manuscript"].update(distinct_from_historical=False),
        "G05_section_order_exact": lambda x: x["manuscript"].update(section_order_exact=False),
        "G06_abstract_contract": lambda x: x["manuscript"]["section_checks"].update(Abstract=False),
        "G07_introduction_contract": lambda x: x["manuscript"]["section_checks"].update(Introduction=False),
        "G08_methods_contract": lambda x: x["manuscript"]["section_checks"].update(Methods=False),
        "G09_results_contract": lambda x: x["manuscript"]["section_checks"].update(Results=False),
        "G10_discussion_contract": lambda x: x["manuscript"]["section_checks"].update(Discussion=False),
        "G11_limitations_contract": lambda x: x["manuscript"]["section_checks"].update(Limitations=False),
        "G12_conclusion_contract": lambda x: x["manuscript"]["section_checks"].update(Conclusion=False),
        "G13_supplement_contract": lambda x: x["manuscript"]["section_checks"].update(Supplement=False),
        "G14_final_claim_ids_exact": lambda x: x["parent_state"]["claim_ids"].pop(),
        "G15_all_claims_placed": lambda x: x["parent_state"]["final_claims"][0].update(paper_placements=[]),
        "G16_required_citations_present": lambda x: x["manuscript"].update(required_citations_present=False),
        "G17_all_citations_resolved": lambda x: x["manuscript"].update(all_citations_resolved=False),
        "G18_figures_exact": lambda x: x["manuscript"]["figures"]["fig:mm-delta"].update(included=False),
        "G19_forbidden_assertions_absent": lambda x: x["manuscript"].update(forbidden_assertive_hits=["the fastest FPGA decoder"]),
        "G20_parent_numbers_exact": lambda x: x["exact_parent_number_checks"].update(mm_zero_headroom=False),
        "G21_nontransferability_explicit": lambda x: x["parent_state"]["nontransfer"].update(global_weighted_score=1.0),
        "G22_board_values_are_null": lambda x: x["parent_state"]["rtl"]["measured_fields"].update(board_latency_ns=222.222),
        "G23_learning_dropped_only": lambda x: x["parent_state"]["learning"].update(primary=True),
        "G24_historical_snapshots_live": lambda x: x["artifact_registry"]["historical_snapshot_01"].update(sha256="f" * 64),
        "G25_source_data_lossless": lambda x: x["source_data"][0].update(payload_sha256="0" * 64),
        "G26_board_task_state_valid": lambda x: x.update(board_state="Blocked"),
        "G27_compiled_pdf_visual_qa": lambda x: x["visual_qa"].update(manual_pass=False),
    }
    cases: list[dict[str, Any]] = []
    for gate, mutate in mutations.items():
        candidate = copy.deepcopy(report)
        mutate(candidate)
        observed = evaluate_gates(candidate)
        cases.append({
            "mutation_id": f"MUT_{gate}",
            "target_gate": gate,
            "rejected": observed.get(gate) is False,
            "failed_gates": [key for key, value in observed.items() if not value],
        })
    return cases


def build_report() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    parents = {key: _load_json(path) for key, path in PARENT_PATHS.items()}
    board_text = (ROOT / "docs/new_task_board.md").read_text(encoding="utf-8")
    parent_state = _parent_snapshot(parents)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": copy.deepcopy(config),
        "artifact_registry": _artifact_registry(config),
        "manuscript": _manuscript_snapshot(config),
        "visual_qa": _visual_qa_snapshot(config),
        "parent_state": parent_state,
        "exact_parent_number_checks": _exact_parent_number_checks(parent_state),
        "board_state": _board_status(board_text, TASK_ID),
        "source_data": [],
    }
    report["source_data"] = _source_rows(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["semantic_mutation_audit"] = {"cases": _semantic_mutations(report)}
    cases = report["semantic_mutation_audit"]["cases"]
    report["semantic_mutation_audit"].update({"detected": sum(item["rejected"] for item in cases), "total": len(cases)})
    all_pass = all(report["gates"].values()) and all(item["rejected"] for item in cases)
    report["verdict"] = VERDICT if all_pass else "FAIL_PHASE6D_MANUSCRIPT_DELTA"
    digest_payload = copy.deepcopy(report)
    digest_payload.pop("generated_at_utc", None)
    report["analysis_sha256"] = _canonical_sha256(digest_payload)
    return report


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    gate = report["gate_summary"]
    mutation = report["semantic_mutation_audit"]
    lines = [
        "# T7.2.6 Phase 6D 双 lane 论文正文 delta",
        "",
        f"- Machine verdict：`{report['verdict']}`。",
        f"- 正文：`{report['manuscript']['path']}`。",
        f"- 门：{gate['passed']}/{gate['total']}；语义 mutation：{mutation['detected']}/{mutation['total']}。",
        f"- Source Data：{len(report['source_data'])} rows，逐行 canonical JSON + SHA-256 可逆。",
        "",
        "## 中心论点",
        "",
        "multimode 软件 lane 的 v1 strongest-baseline headroom 为 0%，因此不建立 frozen-benchmark SOTA；exact single-mode RTL lane 独立建立 six-cycle/II=1、atomic、fail-closed 的 pre-board 贡献。二者只共享 contract bridge，不共享性能分母。",
        "",
        "## 正文消费边界",
        "",
        "- Abstract、Introduction、Methods、Results、Discussion、Limitations、Conclusion 与 Supplement delta 均由逐节 token contract 验证。",
        "- T7.1.5 的 10 条 final claims 原样绑定；multimode negative、board-null、speed prohibition 与 learning dropped 不可删除。",
        "- 旧 51 页稿和 T7.1.1--T7.2.5 保持只读历史快照，不再充当 current manuscript verdict。",
        "- Figure 5 只承载 multimode；Figure 6 只承载 exact single-mode RTL；无 global LER--latency score。",
        "",
        "## 关键数值",
        "",
        "- multimode：strongest baseline 与 causal risk 均 `p_L=0.1119791667`，relative point/LCB=`0%/0%`，pilot/formal/scaling 未访问。",
        "- RTL：17/17 formal gates、21/21 formal mutants、1,000,000 cycles、998,435/998,435 II=1 pairs、0 mismatch，三 seed 最低 Fmax 36.794 MHz。",
        "- 物理板：latency/jitter/deadline/power/transfer/commit 均 null；不声称 fastest 或 SOTA latency。",
        "",
        "## Revocation",
        "",
        "任一父 hash 漂移、删 strongest baseline/0% no-go、填 board-null、把 CNN/student 升为 primary、声称 current RTL 执行 multimode MLD、添加跨 lane 总分或把 post-route estimate 写成 measured，均撤销本合同。",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: Mapping[str, Any]) -> None:
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", DEFAULT_REPORT)
    DEFAULT_SOURCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEFAULT_SOURCE_DATA.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("row_id", "kind", "item_id", "payload_json", "payload_sha256"))
        writer.writeheader()
        writer.writerows(report["source_data"])
    temporary.replace(DEFAULT_SOURCE_DATA)
    _atomic_text(_markdown(report), DEFAULT_MARKDOWN)


def verify_report() -> tuple[bool, dict[str, bool]]:
    if not DEFAULT_REPORT.is_file() or not DEFAULT_SOURCE_DATA.is_file():
        return False, {"outputs_exist": False}
    stored = _load_json(DEFAULT_REPORT)
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    fresh = build_report()
    checks = {
        "outputs_exist": True,
        "stored_verdict": stored.get("verdict") == VERDICT,
        "fresh_verdict": fresh.get("verdict") == VERDICT,
        "stored_analysis_matches": stored.get("analysis_sha256") == fresh.get("analysis_sha256"),
        "source_data_matches": csv_rows == fresh.get("source_data"),
        "all_gates": all(fresh.get("gates", {}).values()),
        "all_mutations": all(item["rejected"] for item in fresh["semantic_mutation_audit"]["cases"]),
    }
    return all(checks.values()), checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        ok, checks = verify_report()
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    report = build_report()
    write_outputs(report)
    print(json.dumps({
        "verdict": report["verdict"],
        "gates": report["gate_summary"],
        "mutations": {
            "detected": report["semantic_mutation_audit"]["detected"],
            "total": report["semantic_mutation_audit"]["total"],
        },
        "source_rows": len(report["source_data"]),
        "analysis_sha256": report["analysis_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
