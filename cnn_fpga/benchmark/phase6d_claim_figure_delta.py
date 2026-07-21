"""Freeze the T7.1.5 Phase-6D claim/figure delta without rewriting T7.1.1--T7.1.4."""

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

from cnn_fpga.benchmark import claim_evidence_boundary_matrix as snapshot_claims
from cnn_fpga.benchmark import main_figure_contract as snapshot_figures_1_2
from cnn_fpga.benchmark import main_result_figure_contract as snapshot_figures_3_4
from cnn_fpga.benchmark import phase6d_dual_lane_evidence_matrix as matrix_gate
from cnn_fpga.benchmark import phase6d_final_dual_lane_gate as final_gate
from cnn_fpga.benchmark import supplement_figure_contract as snapshot_supplement


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.1.5"
SCHEMA_VERSION = "t7.1.5-phase6d-claim-figure-delta-v1"
VERDICT = "PASS_PHASE6D_CLAIM_FIGURE_DELTA_RTL_ONLY"

CONFIG = ROOT / "configs/phase6d/t7_1_5_claim_figure_delta.json"
BOARD = ROOT / "docs/new_task_board.md"
REPORT = ROOT / "docs/t7_1_5_phase6d_claim_figure_delta.json"
SOURCE_DATA = ROOT / "docs/t7_1_5_phase6d_claim_figure_delta_source_data.csv"
MARKDOWN = ROOT / "docs/phase6d_claim_figure_delta.md"
RENDERER = ROOT / "docs/figures/make_t7_1_5_phase6d_delta.py"
FIGURE_DIR = ROOT / "docs/figures/t7_1_5_phase6d_delta"
MANIFEST = FIGURE_DIR / "figure_manifest.json"

MATRIX_REPORT = ROOT / "docs/t6_26_3_dual_lane_evidence_matrix.json"
FINAL_REPORT = ROOT / "docs/t6_26_4_final_dual_lane_gate.json"

EVIDENCE_CATEGORIES = ("reports", "raw_data", "configs", "code", "sources")
MM_LANE = "MULTIMODE_SOFTWARE_ALGORITHM"
RTL_LANE = "SINGLE_MODE_DETERMINISTIC_RTL"
LEARNING_LANE = "LEARNED_APPROXIMATION_EXTENSION"


class IntegrityError(RuntimeError):
    """Raised when a frozen claim/figure contract is internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"not a JSON object: {path}")
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
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _task_statuses(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for task, status in re.findall(r"^\| (T[^| ]+) \| ([^|]+) \|", text, flags=re.MULTILINE):
        result.setdefault(task.strip(), status.strip())
    return result


def _normalise_status(task: str, status: str) -> str:
    if task == "T7.1.5" and status in {"Todo", "In Progress", "Done"}:
        return "TODO_ACTIVE_OR_DONE"
    return status


def _parent_verification() -> dict[str, bool]:
    calls = {
        "phase6d_matrix": matrix_gate.verify,
        "phase6d_final_gate": final_gate.verify,
        "snapshot_t7_1_1": snapshot_claims.verify_report,
        "snapshot_t7_1_2_contract": snapshot_figures_1_2.verify_report,
        "snapshot_t7_1_2_bundle": snapshot_figures_1_2.verify_bundle,
        "snapshot_t7_1_3_contract": snapshot_figures_3_4.verify_report,
        "snapshot_t7_1_3_bundle": snapshot_figures_3_4.verify_bundle,
        "snapshot_t7_1_4_contract": snapshot_supplement.verify_report,
        "snapshot_t7_1_4_bundle": snapshot_supplement.verify_bundle,
    }
    result: dict[str, bool] = {}
    for name, call in calls.items():
        try:
            call()
            result[name] = True
        except Exception:
            result[name] = False
    return result


def _artifact_registry(config: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for artifact_id, frozen in matrix["artifact_registry"].items():
        path = ROOT / str(frozen["path"])
        registry[artifact_id] = _binding(path)
    additions = {
        "matrix_contract_report": MATRIX_REPORT,
        "matrix_contract_source_data": ROOT / "docs/t6_26_3_dual_lane_evidence_source_data.csv",
        "matrix_contract_markdown": ROOT / "docs/phase6d_dual_lane_evidence_matrix.md",
        "final_gate_report": FINAL_REPORT,
        "final_gate_source_data": ROOT / "docs/t6_26_4_final_dual_lane_gate_source_data.csv",
        "final_gate_config": ROOT / "configs/phase6d/t6_26_4_final_dual_lane_gate.json",
        "final_gate_code": ROOT / "cnn_fpga/benchmark/phase6d_final_dual_lane_gate.py",
        "final_gate_markdown": ROOT / "docs/phase6d_final_dual_lane_gate.md",
        "delta_config": CONFIG,
        "delta_contract_code": Path(__file__).resolve(),
        "delta_renderer": RENDERER,
        "task_board": BOARD,
    }
    for artifact_id, path in additions.items():
        registry[artifact_id] = _binding(path)
    for index, relative_path in enumerate(config["historical_snapshot_paths"], start=1):
        registry[f"historical_snapshot_{index:02d}"] = _binding(ROOT / relative_path)
    return registry


def _augment_evidence(evidence: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(dict(evidence))
    result["reports"] = list(dict.fromkeys([*result.get("reports", []), "matrix_contract_report", "final_gate_report"]))
    result["configs"] = list(dict.fromkeys([*result.get("configs", []), "delta_config"]))
    result["code"] = list(dict.fromkeys([*result.get("code", []), "delta_contract_code", "delta_renderer"]))
    result["selectors"] = list(dict.fromkeys([*result.get("selectors", []), "T7.1.5.figure_contract", "T7.1.5.revocation_conditions"]))
    ids = [artifact_id for category in EVIDENCE_CATEGORIES for artifact_id in result.get(category, [])]
    _require(all(artifact_id in artifacts for artifact_id in ids), "element references an unknown artifact")
    result["hashes"] = {artifact_id: artifacts[artifact_id]["sha256"] for artifact_id in ids}
    return result


def _claim_index(final: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["claim_id"]: row for row in final["final_claims"]}


def _build_elements(
    config: Mapping[str, Any],
    matrix: Mapping[str, Any],
    final: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parent = {row["element_id"]: row for row in matrix["figure_contract"]["elements"]}
    claims = _claim_index(final)
    elements: list[dict[str, Any]] = []
    for element_id, placement in config["parent_element_layout"].items():
        source = parent[element_id]
        claim_ids = list(config["element_claims"][element_id])
        revocations = sorted({item for claim_id in claim_ids for item in claims[claim_id]["revocation_conditions"]})
        elements.append({
            "element_id": element_id,
            "source_element_id": element_id,
            "source_element_payload_sha256": _canonical_sha256(source),
            "figure_id": placement[0],
            "panel": placement[1],
            "lane_id": source["lane_id"],
            "metric_namespace": source["metric_namespace"],
            "title": source["title"],
            "status": "PARENT_EVIDENCE_RETAINED",
            "value": copy.deepcopy(source["value"]),
            "allowed_wording": source["allowed_wording"],
            "forbidden_interpretation": list(source["forbidden_interpretation"]),
            "claim_ids": claim_ids,
            "revocation_conditions": revocations,
            "evidence": _augment_evidence(source["evidence"], artifacts),
        })
    for element_id, spec in config["derived_elements"].items():
        template = parent[spec["evidence_template"]]
        claim_ids = list(spec["claim_ids"])
        revocations = sorted({item for claim_id in claim_ids for item in claims[claim_id]["revocation_conditions"]})
        elements.append({
            "element_id": element_id,
            "source_element_id": None,
            "source_element_payload_sha256": None,
            "figure_id": spec["figure_id"],
            "panel": spec["panel"],
            "lane_id": spec["lane_id"],
            "metric_namespace": spec["metric_namespace"],
            "title": spec["title"],
            "status": spec["status"],
            "value": copy.deepcopy(spec["value"]),
            "allowed_wording": spec["allowed_wording"],
            "forbidden_interpretation": list(spec["forbidden_interpretation"]),
            "claim_ids": claim_ids,
            "revocation_conditions": revocations,
            "evidence": _augment_evidence(template["evidence"], artifacts),
        })
    return elements


def _build_claims(
    config: Mapping[str, Any],
    final: Mapping[str, Any],
    elements: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    element_map: dict[str, list[str]] = {}
    for element in elements:
        for claim_id in element["claim_ids"]:
            element_map.setdefault(claim_id, []).append(element["element_id"])
    result: list[dict[str, Any]] = []
    for source in final["final_claims"]:
        claim_id = source["claim_id"]
        evidence = _augment_evidence(source["parent_evidence_keys"], artifacts)
        value = copy.deepcopy(dict(source))
        value.update({
            "figure_ids": list(config["claim_figure_map"][claim_id]),
            "element_ids": sorted(element_map.get(claim_id, [])),
            "render_binding": "bundle_caption_only" if claim_id == "DUAL_LANE_NONTRANSFERABILITY" else "figure_element",
            "evidence": evidence,
            "source_final_claim_payload_sha256": _canonical_sha256(source),
        })
        result.append(value)
    return result


FIGURE_5_CAPTION = (
    "Fig. 5 | Multimode software evidence remains below the frozen-benchmark promotion gate. "
    "a, Train-only causal headroom compares static-mixture exact MLD with the proposed risk action over 79,872 rounds; "
    "both have pL=0.111979 and the paired relative point estimate and 95% lower confidence bound are 0%, so the v1 branch is NO-GO. "
    "b, The opened d=3 task-local study (9.6 million cycles, 32 seed clusters and 20,000 bootstrap resamples) is retained only as context for LER, "
    "non-overlapping 512-cycle worst-window/CVaR95 tail and measured host runtime; it does not use the strongest denominator. "
    "c, Pilot, formal and frozen-benchmark scaling results were not accessed after the headroom stop. "
    "d, CNN/student is absent from the primary result and appears only as a dropped ablation status. "
    "No RTL timing or hardware claim is inferred from this figure. Source data are provided as a Source Data file."
)

FIGURE_6_CAPTION = (
    "Fig. 6 | Exact single-mode deterministic and fail-closed pre-board RTL evidence. "
    "a, The converged fast path has a six-cycle source-to-action contract and initiation interval one. "
    "b, CRC/version admission, inactive-bank staging, safe-boundary atomic commit and last-known-good recovery implement the stated fail-closed transaction. "
    "c, Seventeen of seventeen formal gates and 21 of 21 formal mutants pass; an independent one-million-cycle CXXRTL run compares the full 148-byte public vector with zero mismatch. "
    "d,e, Three GW2AR seeds pass 27 MHz (minimum 36.794 MHz), with resource utilization reported for the whole-harness observability top; all critical paths end in the observability fold, so Fmax is not a bare-core result. "
    "f, Board latency, jitter, deadline misses, power and physical transfer/commit latency remain null. "
    "These data establish neither measured hardware performance nor a speed ranking, and the current RTL does not execute the multimode decoder. "
    "Source data are provided as a Source Data file."
)


def _build_figures(config: Mapping[str, Any]) -> dict[str, Any]:
    figures = copy.deepcopy(config["figure_contracts"])
    figures["Figure 5"]["caption"] = FIGURE_5_CAPTION
    figures["Figure 5"]["evidence_hierarchy"] = {
        "hero": "strongest-baseline LER and zero causal headroom",
        "validation": "opened task-local LER/tail/compute context",
        "controls": "pilot/formal/scaling not run and learning dropped",
    }
    figures["Figure 5"]["reviewer_risks"] = [
        "opened task-local gain mistaken for strongest-baseline SOTA",
        "unaccessed pilot/formal/scaling shown as zero rather than unavailable",
        "CNN/student promoted from dropped inset",
    ]
    figures["Figure 6"]["caption"] = FIGURE_6_CAPTION
    figures["Figure 6"]["evidence_hierarchy"] = {
        "hero": "six-cycle II=1 atomic/fail-closed transaction",
        "validation": "formal plus million-cycle exact CXXRTL qualification",
        "controls": "whole-harness P&R caveat, resources and board-null",
    }
    figures["Figure 6"]["reviewer_risks"] = [
        "whole-harness Fmax mistaken for bare-core or board latency",
        "post-route estimate mistaken for measured hardware",
        "single-mode RTL mistaken for a multimode decoder deployment",
    ]
    return figures


SOURCE_FIELDS = (
    "record_type", "record_id", "figure_id", "panel", "lane_id", "metric_namespace",
    "status", "payload_json", "payload_sha256", "evidence_ids_json", "evidence_hashes_json",
    "claim_ids_json", "revocation_conditions_json",
)


def _source_row(record_type: str, record_id: str, payload: Any, **columns: Any) -> dict[str, str]:
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    row = {field: "" for field in SOURCE_FIELDS}
    row.update({
        "record_type": record_type,
        "record_id": record_id,
        "payload_json": payload_json,
        "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
    })
    for key, value in columns.items():
        row[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return row


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for figure_id, figure in report["figures"].items():
        rows.append(_source_row("figure_contract", figure_id, figure, figure_id=figure_id, status="FROZEN_DELTA"))
    for element in report["elements"]:
        evidence_ids = [artifact_id for category in EVIDENCE_CATEGORIES for artifact_id in element["evidence"][category]]
        rows.append(_source_row(
            "figure_element", element["element_id"], element,
            figure_id=element["figure_id"], panel=element["panel"], lane_id=element["lane_id"],
            metric_namespace=element["metric_namespace"], status=element["status"],
            evidence_ids_json=evidence_ids, evidence_hashes_json=element["evidence"]["hashes"],
            claim_ids_json=element["claim_ids"], revocation_conditions_json=element["revocation_conditions"],
        ))
    for claim in report["claims"]:
        evidence_ids = [artifact_id for category in EVIDENCE_CATEGORIES for artifact_id in claim["evidence"][category]]
        rows.append(_source_row(
            "claim", claim["claim_id"], claim,
            figure_id=",".join(claim["figure_ids"]), lane_id=claim["lane_id"], status=claim["final_disposition"],
            evidence_ids_json=evidence_ids, evidence_hashes_json=claim["evidence"]["hashes"],
            claim_ids_json=[claim["claim_id"]], revocation_conditions_json=claim["revocation_conditions"],
        ))
    for snapshot_id, binding in report["historical_snapshots"].items():
        rows.append(_source_row("historical_snapshot", snapshot_id, binding, status="READ_ONLY_PRESERVED"))
    for artifact_id, binding in report["artifact_registry"].items():
        rows.append(_source_row("artifact", artifact_id, binding, status="LIVE_HASH_BOUND"))
    return rows


def _write_source_data(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_source_data(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _render_markdown(report: Mapping[str, Any]) -> str:
    figure5 = report["figures"]["Figure 5"]
    figure6 = report["figures"]["Figure 6"]
    return "\n".join([
        "# T7.1.5 Phase 6D 双 lane claim/figure delta", "",
        "## Figure contract", "",
        "- Backend：Python / matplotlib only。",
        "- 历史策略：T7.1.1--T7.1.4 只读保留；本任务只新增 Figure 5--6，不覆盖、改色或重命名旧图。",
        "- Bundle 边界：两图无跨 lane 箭头、无共同性能轴、无 LER--latency 加权总分。",
        f"- Figure 5 核心结论：{figure5['core_conclusion']}",
        f"- Figure 6 核心结论：{figure6['core_conclusion']}", "",
        "## Panel map", "",
        "### Figure 5 — multimode software only", "",
        *[f"- {panel}：{label}。" for panel, label in figure5["panel_map"].items()], "",
        "### Figure 6 — exact single-mode RTL only", "",
        *[f"- {panel}：{label}。" for panel, label in figure6["panel_map"].items()], "",
        "## Evidence hierarchy and review risks", "",
        "- Figure 5 hero 是 strongest-baseline 零 headroom；opened LER/tail/compute 只作 context，未执行的 scaling/pilot/formal 不填零。",
        "- Figure 6 hero 是 cycle/transaction contract；formal/CXXRTL 是 pre-board validation，P&R 是 whole-harness estimate，board 字段保持 null。",
        "- CNN/student 只在 Figure 5d 以 dropped/ablation 状态出现，不进入任一 primary verdict。",
        "- current RTL 不执行 multimode MLD；single-mode timing 不能替 Figure 5 补门，Figure 5 的 LER 也不能证明 Figure 6 的硬件实现。", "",
        "## Figure legends", "",
        figure5["caption"], "",
        figure6["caption"], "",
        "## Revocation rule", "",
        "任一 raw/config/code/source hash 漂移、任一父门失败、删除 multimode NO-GO、填充 board-null、把 learning 提升为 primary、写入 global score、复用旧图输出路径或删掉 whole-harness caveat，均撤销本 delta。", "",
        f"Machine verdict: `{report['verdict']}`。", "",
    ])


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w%=-]+\b", text))


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    config = _load(CONFIG)
    matrix = _load(MATRIX_REPORT)
    final = _load(FINAL_REPORT)
    elements = {row["element_id"]: row for row in report["elements"]}
    claims = {row["claim_id"]: row for row in report["claims"]}
    parent_elements = {row["element_id"]: row for row in matrix["figure_contract"]["elements"]}
    parent_claims = {row["claim_id"]: row for row in final["final_claims"]}
    artifacts = report["artifact_registry"]
    current_parent = _parent_verification() if check_live_files else report["parent_verification"]
    current_statuses = _task_statuses(BOARD.read_text(encoding="utf-8")) if check_live_files else report["board_snapshot"]["raw_statuses"]
    board_norm = {task: _normalise_status(task, current_statuses.get(task, "MISSING")) for task in report["board_snapshot"]["normalised_statuses"]}
    evidence_complete = True
    hashes_match = True
    for row in [*elements.values(), *claims.values()]:
        evidence = row["evidence"]
        evidence_complete &= all(bool(evidence.get(category)) for category in EVIDENCE_CATEGORIES)
        ids = [artifact_id for category in EVIDENCE_CATEGORIES for artifact_id in evidence[category]]
        hashes_match &= set(ids) == set(evidence["hashes"])
        hashes_match &= all(artifact_id in artifacts and evidence["hashes"][artifact_id] == artifacts[artifact_id]["sha256"] for artifact_id in ids)
    figure5_lanes = {row["lane_id"] for row in elements.values() if row["figure_id"] == "Figure 5"}
    figure6_lanes = {row["lane_id"] for row in elements.values() if row["figure_id"] == "Figure 6"}
    namespaces = {
        figure_id: {row["metric_namespace"] for row in elements.values() if row["figure_id"] == figure_id}
        for figure_id in report["figures"]
    }
    actual_rows = _read_source_data(ROOT / report["source_data"]["path"])
    expected_rows = _source_rows(report)
    captions = {figure_id: figure["caption"] for figure_id, figure in report["figures"].items()}
    mm_value = elements["MM-E4"]["value"]
    rtl_latency = elements["RTL-E1"]["value"]
    board_values = elements["RTL-E6"]["value"]
    return {
        "G01_all_phase6d_and_historical_parent_verifiers_pass_live": all(current_parent.values()) and report["parent_verification"] == current_parent and report["parent_verdicts"] == config["expected_parent_verdicts"],
        "G02_artifact_registry_and_embedded_hashes_are_complete_and_live": all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in artifacts.values()) and hashes_match and (not check_live_files or all(_live(row) for row in artifacts.values())),
        "G03_t7_1_1_to_t7_1_4_snapshots_are_read_only_live_and_exact": len(report["historical_snapshots"]) == len(config["historical_snapshot_paths"]) == 11 and [row["path"] for row in report["historical_snapshots"].values()] == config["historical_snapshot_paths"] and (not check_live_files or all(_live(row) for row in report["historical_snapshots"].values())),
        "G04_two_new_figures_have_complete_contracts_and_new_numbering": set(report["figures"]) == {"Figure 5", "Figure 6"} and all(row["core_conclusion"] and row["panel_map"] and row["width_mm"] == 183 and row["archetype"] in {"quantitative grid", "schematic-led composite"} for row in report["figures"].values()),
        "G05_all_parent_elements_are_consumed_once_plus_explicit_scaling_null": set(elements) == {*parent_elements, "MM-D6"} and len(elements) == len(parent_elements) + 1 and all(
            elements[element_id]["source_element_payload_sha256"] == _canonical_sha256(parent_elements[element_id])
            and elements[element_id]["figure_id"] == config["parent_element_layout"][element_id][0]
            and elements[element_id]["panel"] == config["parent_element_layout"][element_id][1]
            and elements[element_id]["claim_ids"] == config["element_claims"][element_id]
            and all(elements[element_id][key] == parent_elements[element_id][key] for key in ("lane_id", "metric_namespace", "title", "value", "allowed_wording", "forbidden_interpretation"))
            for element_id in parent_elements
        ),
        "G06_required_metric_namespaces_are_complete_without_numeric_scaling_fabrication": all(namespaces[figure_id] == set(required) for figure_id, required in config["required_metric_namespaces"].items()) and elements["MM-D6"]["metric_namespace"] == "SCALING",
        "G07_figure_lanes_are_separate_and_learning_is_only_a_figure5_inset": figure5_lanes == {MM_LANE, LEARNING_LANE} and figure6_lanes == {RTL_LANE} and elements["ML-E1"]["figure_id"] == "Figure 5" and elements["ML-E1"]["panel"] == "d",
        "G08_no_cross_lane_edge_global_score_or_common_performance_denominator": report["bundle_boundary"] == config["forbidden_bundle_properties"] and report["bundle_boundary"]["global_weighted_score"] is None and report["bundle_boundary"]["cross_lane_visual_edges"] == 0,
        "G09_all_ten_final_claims_preserve_payload_wording_gaps_revocation_and_placement": set(claims) == set(parent_claims) and len(claims) == 10 and all(
            claim["source_final_claim_payload_sha256"] == _canonical_sha256(parent_claims[claim_id])
            and all(claim[key] == parent_claims[claim_id][key] for key in parent_claims[claim_id])
            and claim["figure_ids"] == config["claim_figure_map"][claim_id]
            and (bool(claim["element_ids"]) if claim_id != "DUAL_LANE_NONTRANSFERABILITY" else claim["element_ids"] == [])
            and claim["render_binding"] == ("bundle_caption_only" if claim_id == "DUAL_LANE_NONTRANSFERABILITY" else "figure_element")
            and claim["final_wording"] and claim["forbidden_wording"] and claim["blocking_gaps"] and claim["revocation_conditions"] and claim["paper_placements"]
            for claim_id, claim in claims.items()
        ),
        "G10_every_claim_and_element_has_report_raw_config_code_source_selector_hash_and_revocation": evidence_complete and hashes_match
        and all(row["evidence"]["selectors"] and row["revocation_conditions"] for row in elements.values())
        and all(row["evidence"]["selectors"] and row["revocation_conditions"] for row in claims.values())
        and all(elements[element_id]["evidence"] == _augment_evidence(parent_elements[element_id]["evidence"], artifacts) for element_id in parent_elements)
        and elements["MM-D6"]["evidence"] == _augment_evidence(parent_elements[config["derived_elements"]["MM-D6"]["evidence_template"]]["evidence"], artifacts)
        and all(claims[claim_id]["evidence"] == _augment_evidence(parent_claims[claim_id]["parent_evidence_keys"], artifacts) for claim_id in parent_claims),
        "G11_multimode_strongest_baseline_values_and_no_go_are_exact": mm_value == parent_elements["MM-E4"]["value"] and mm_value["baseline_p_L"] == mm_value["proposed_p_L"] and mm_value["relative_improvement_point"] == mm_value["relative_improvement_lcb"] == 0.0 and final["lane_decisions"][MM_LANE]["decision"] == "NO_GO",
        "G12_multimode_pilot_formal_scaling_and_sota_remain_unavailable": elements["MM-E5"]["value"] is None and elements["MM-D6"]["value"] == {"distance_scaling": None, "sigma_scaling": None, "pilot_accessed": False, "formal_accessed": False} and not report["bundle_boundary"].get("multimode_frozen_benchmark_sota", False),
        "G13_rtl_six_cycle_ii1_atomic_longrun_and_board_null_are_exact": rtl_latency == {"cycles": 6, "II": 1} and elements["RTL-E2"]["value"] == parent_elements["RTL-E2"]["value"] and elements["RTL-E3"]["value"]["cycles_qualified"] == 1_000_000 and elements["RTL-E3"]["value"]["mismatches"] == 0 and board_values == parent_elements["RTL-E6"]["value"] and all(value is None for value in board_values.values()),
        "G14_postroute_is_whole_harness_only_and_speed_claim_is_prohibited": all(row["wrapper_may_dominate"] for row in elements["RTL-E4"]["value"]["critical_paths"]) and "bare-core Fmax" in elements["RTL-E4"]["forbidden_interpretation"] and claims["RTL_SPEED_ADVANTAGE_PROHIBITED"]["final_disposition"] == "PROHIBITED_POSITIVE",
        "G15_learning_is_dropped_absent_and_cannot_change_any_verdict": elements["ML-E1"]["value"] == {"T6.26.1": "Dropped", "T6.26.2": "Dropped", "present_in_primary_rtl": False} and claims["LEARNING_APPROXIMATION_DROPPED"]["final_disposition"] == "DROPPED_ABLATION_ONLY" and final["lane_decisions"][LEARNING_LANE]["direct_evidence"]["changes_overall_verdict"] is False,
        "G16_captions_are_separate_self_contained_and_bounded": _word_count(captions["Figure 5"]) <= 300 and _word_count(captions["Figure 6"]) <= 300 and all(token in captions["Figure 5"] for token in ("NO-GO", "not accessed", "dropped ablation", "No RTL timing")) and all(token in captions["Figure 6"] for token in ("six-cycle", "one-million-cycle", "whole", "remain null", "does not execute the multimode decoder")) and "pL=0.111979" not in captions["Figure 6"] and "36.794 MHz" not in captions["Figure 5"],
        "G17_export_contract_is_python_only_editable_multiformat_and_new_path": report["export_contract"] == config["export_contract"] and report["output_files"] == config["output_files"] and all(path.startswith(("figure5_", "figure6_")) for path in report["output_files"]) and not any(path in json.dumps(report["historical_snapshots"]) for path in report["output_files"]),
        "G18_source_data_is_exact_lossless_and_row_hash_recomputed": actual_rows == expected_rows and report["source_data"]["rows"] == len(expected_rows) and report["source_data"]["record_type_counts"] == {kind: sum(row["record_type"] == kind for row in expected_rows) for kind in sorted({row["record_type"] for row in expected_rows})} and all(row["payload_sha256"] == hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest() for row in actual_rows) and (not check_live_files or _live(report["source_data"])),
        "G19_human_contract_is_live_and_contains_both_legends_and_boundaries": (ROOT / report["markdown"]["path"]).is_file() and all(token in (ROOT / report["markdown"]["path"]).read_text(encoding="utf-8") for token in ("Figure 5 — multimode software only", "Figure 6 — exact single-mode RTL only", "无 LER--latency 加权总分", "board 字段保持 null", "current RTL 不执行 multimode MLD", "Source data are provided")) and (not check_live_files or _live(report["markdown"])),
        "G20_one_independent_semantic_mutation_targets_every_gate": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 22 and len(report["semantic_mutation_audit"]["cases"]) == 22,
        "G21_delta_config_contract_renderer_board_and_parent_reports_are_bound": {"delta_config", "delta_contract_code", "delta_renderer", "task_board", "matrix_contract_report", "final_gate_report"} <= set(artifacts),
        "G22_board_status_transition_preserves_snapshots_and_accepts_active_or_done_delta": report["board_snapshot"]["normalised_statuses"] == board_norm and all(board_norm[f"T7.1.{index}"] == "Done" for index in range(1, 5)) and board_norm["T7.1.5"] == "TODO_ACTIVE_OR_DONE" and board_norm["T6.9.2"] == "Blocked",
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def element(value: dict[str, Any], element_id: str) -> dict[str, Any]:
        return next(row for row in value["elements"] if row["element_id"] == element_id)

    def claim(value: dict[str, Any], claim_id: str) -> dict[str, Any]:
        return next(row for row in value["claims"] if row["claim_id"] == claim_id)

    def attempt(name: str, target: str, change: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 22, "detected": 22, "cases": []}
        change(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[target]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": target, "rejected": rejected})

    attempt("forge_parent_pass", "G01_all_phase6d_and_historical_parent_verifiers_pass_live", lambda x: x["parent_verification"].update(phase6d_final_gate=False))
    attempt("forge_artifact_hash", "G02_artifact_registry_and_embedded_hashes_are_complete_and_live", lambda x: x["artifact_registry"]["delta_config"].update(sha256="0"))
    attempt("drop_historical_snapshot", "G03_t7_1_1_to_t7_1_4_snapshots_are_read_only_live_and_exact", lambda x: x["historical_snapshots"].pop("historical_snapshot_11"))
    attempt("rename_new_figure_as_old", "G04_two_new_figures_have_complete_contracts_and_new_numbering", lambda x: x["figures"].update({"Figure 4": x["figures"].pop("Figure 5")}))
    attempt("drop_parent_element", "G05_all_parent_elements_are_consumed_once_plus_explicit_scaling_null", lambda x: x["elements"].pop())
    attempt("hide_scaling_namespace", "G06_required_metric_namespaces_are_complete_without_numeric_scaling_fabrication", lambda x: element(x, "MM-D6").update(metric_namespace="EVIDENCE_STATE"))
    attempt("move_rtl_into_multimode", "G07_figure_lanes_are_separate_and_learning_is_only_a_figure5_inset", lambda x: element(x, "RTL-E1").update(figure_id="Figure 5"))
    attempt("add_global_score", "G08_no_cross_lane_edge_global_score_or_common_performance_denominator", lambda x: x["bundle_boundary"].update(global_weighted_score=1.0))
    attempt("drop_claim_revocation", "G09_all_ten_final_claims_preserve_payload_wording_gaps_revocation_and_placement", lambda x: claim(x, "MM_V1_CAUSAL_HEADROOM_NO_GO").update(revocation_conditions=[]))
    attempt("remove_raw_binding", "G10_every_claim_and_element_has_report_raw_config_code_source_selector_hash_and_revocation", lambda x: element(x, "RTL-E2")["evidence"].update(raw_data=[]))
    attempt("forge_multimode_gain", "G11_multimode_strongest_baseline_values_and_no_go_are_exact", lambda x: element(x, "MM-E4")["value"].update(proposed_p_L=0.09))
    attempt("fabricate_scaling", "G12_multimode_pilot_formal_scaling_and_sota_remain_unavailable", lambda x: element(x, "MM-D6")["value"].update(distance_scaling=0.2))
    attempt("shorten_rtl_latency", "G13_rtl_six_cycle_ii1_atomic_longrun_and_board_null_are_exact", lambda x: element(x, "RTL-E1")["value"].update(cycles=5))
    attempt("rename_harness_as_bare_core", "G14_postroute_is_whole_harness_only_and_speed_claim_is_prohibited", lambda x: element(x, "RTL-E4").update(forbidden_interpretation=[]))
    attempt("promote_learning", "G15_learning_is_dropped_absent_and_cannot_change_any_verdict", lambda x: claim(x, "LEARNING_APPROXIMATION_DROPPED").update(final_disposition="PRIMARY"))
    attempt("cross_contaminate_caption", "G16_captions_are_separate_self_contained_and_bounded", lambda x: x["figures"]["Figure 5"].update(caption=x["figures"]["Figure 5"]["caption"] + " 36.794 MHz"))
    attempt("switch_backend", "G17_export_contract_is_python_only_editable_multiformat_and_new_path", lambda x: x["export_contract"].update(backend="R"))
    attempt("forge_source_row_count", "G18_source_data_is_exact_lossless_and_row_hash_recomputed", lambda x: x["source_data"].update(rows=0))
    attempt("disconnect_markdown", "G19_human_contract_is_live_and_contains_both_legends_and_boundaries", lambda x: x["markdown"].update(path="docs/not_found.md"))
    attempt("forge_mutation_count", "G20_one_independent_semantic_mutation_targets_every_gate", lambda x: x.update(semantic_mutation_audit={"count": 22, "detected": 21, "cases": []}))
    attempt("drop_renderer_binding", "G21_delta_config_contract_renderer_board_and_parent_reports_are_bound", lambda x: x["artifact_registry"].pop("delta_renderer"))
    attempt("overwrite_snapshot_status", "G22_board_status_transition_preserves_snapshots_and_accepts_active_or_done_delta", lambda x: x["board_snapshot"]["normalised_statuses"].update({"T7.1.4": "In Progress"}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report(source_data: Path = SOURCE_DATA, markdown: Path = MARKDOWN) -> dict[str, Any]:
    config = _load(CONFIG)
    matrix = _load(MATRIX_REPORT)
    final = _load(FINAL_REPORT)
    artifacts = _artifact_registry(config, matrix)
    elements = _build_elements(config, matrix, final, artifacts)
    claims = _build_claims(config, final, elements, artifacts)
    statuses = _task_statuses(BOARD.read_text(encoding="utf-8"))
    status_tasks = ["T6.9.2", "T7.1.1", "T7.1.2", "T7.1.3", "T7.1.4", "T7.1.5"]
    historical = {
        f"historical_snapshot_{index:02d}": artifacts[f"historical_snapshot_{index:02d}"]
        for index in range(1, len(config["historical_snapshot_paths"]) + 1)
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_verification": _parent_verification(),
        "parent_verdicts": {"matrix": matrix["verdict"], "final_gate": final["verdict"]},
        "board_snapshot": {
            "raw_statuses": {task: statuses.get(task, "MISSING") for task in status_tasks},
            "normalised_statuses": {task: _normalise_status(task, statuses.get(task, "MISSING")) for task in status_tasks},
        },
        "historical_snapshot_policy": "PRESERVE_T7_1_1_TO_T7_1_4_AND_ADD_NEW_FIGURES_5_6",
        "historical_snapshots": historical,
        "artifact_registry": artifacts,
        "figures": _build_figures(config),
        "elements": elements,
        "claims": claims,
        "bundle_boundary": copy.deepcopy(config["forbidden_bundle_properties"]),
        "export_contract": copy.deepcopy(config["export_contract"]),
        "output_files": list(config["output_files"]),
        "source_data": {"path": _relative(source_data), "sha256": "", "bytes": 0, "rows": 0, "record_type_counts": {}},
        "markdown": {"path": _relative(markdown), "sha256": "", "bytes": 0},
        "semantic_mutation_audit": {"count": 22, "detected": 22, "cases": []},
        "verdict": VERDICT,
    }
    rows = _source_rows(report)
    _write_source_data(rows, source_data)
    counts = {kind: sum(row["record_type"] == kind for row in rows) for kind in sorted({row["record_type"] for row in rows})}
    report["source_data"] = {**_binding(source_data), "rows": len(rows), "record_type_counts": counts}
    _atomic_text(markdown, _render_markdown(report))
    report["markdown"] = _binding(markdown)
    # Rebuild Source Data after markdown binding is fixed; rows intentionally exclude generated markdown/source bindings.
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
        "failed": [gate for gate, passed in report["gates"].items() if not passed],
    }
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_PHASE6D_CLAIM_FIGURE_DELTA"
    report["analysis_sha256"] = _canonical_sha256({
        key: report[key]
        for key in (
            "parent_verification", "parent_verdicts", "board_snapshot", "historical_snapshot_policy",
            "historical_snapshots", "artifact_registry", "figures", "elements", "claims", "bundle_boundary",
            "export_contract", "output_files", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict",
        )
    })
    return report


def verify(report: Mapping[str, Any] | None = None, path: Path = REPORT) -> dict[str, Any]:
    value = dict(report) if report is not None else _load(path)
    gates = evaluate_gates(value)
    expected_hash = _canonical_sha256({
        key: value[key]
        for key in (
            "parent_verification", "parent_verdicts", "board_snapshot", "historical_snapshot_policy",
            "historical_snapshots", "artifact_registry", "figures", "elements", "claims", "bundle_boundary",
            "export_contract", "output_files", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict",
        )
    })
    checks = {
        "identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION,
        "gates": value.get("gates") == gates and all(gates.values()),
        "verdict": value.get("verdict") == VERDICT,
        "analysis_hash": value.get("analysis_sha256") == expected_hash,
    }
    _require(all(checks.values()), f"T7.1.5 verification failed: {[name for name, passed in checks.items() if not passed]}")
    return {
        "verdict": value["verdict"],
        "gates": value["gate_summary"],
        "mutations": {
            "detected": value["semantic_mutation_audit"]["detected"],
            "total": value["semantic_mutation_audit"]["count"],
        },
        "source_rows": value["source_data"]["rows"],
        "analysis_sha256": value["analysis_sha256"],
    }


def _historical_output_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in (
        ROOT / "docs/figures/t7_1_2_main_figures/figure_manifest.json",
        ROOT / "docs/figures/t7_1_3_main_figures/figure_manifest.json",
        ROOT / "docs/figures/t7_1_4_supplement_figures/figure_manifest.json",
    ):
        manifest = _load(path)
        hashes.update(row["sha256"] for row in manifest.get("outputs", {}).values())
    return hashes


def verify_bundle(manifest_path: Path = MANIFEST) -> dict[str, bool]:
    report = _load(REPORT)
    verify(report)
    manifest = _load(manifest_path)
    outputs = manifest.get("outputs", {})
    qa = manifest.get("qa", {})
    historical_hashes = _historical_output_hashes()
    required_tokens = {
        "figure5_multimode_software_delta.svg": ["NO-GO", "0.0%", "Pilot", "CNN / student"],
        "figure6_single_mode_rtl_delta.svg": ["6 cycles", "II=1", "1,000,000", "BOARD: UNMEASURED"],
    }
    checks = {
        "identity": manifest.get("task_id") == TASK_ID and manifest.get("schema_version") == "t7.1.5-phase6d-figure-bundle-v1",
        "backend": manifest.get("backend") == "Python/matplotlib only" and qa.get("backend_exclusive") is True,
        "contract_live": manifest.get("contract") == _binding(REPORT),
        "source_data_live": manifest.get("source_data") == _binding(SOURCE_DATA),
        "renderer_live": manifest.get("renderer") == _binding(RENDERER),
        "outputs_exact_and_live": set(outputs) == set(report["output_files"]) and all(_live(row) for row in outputs.values()),
        "outputs_are_new_not_snapshot_copies": not ({row["sha256"] for row in outputs.values()} & historical_hashes),
        "editable_vector": all(qa.get("svg_text_nodes", {}).get(name, 0) >= 25 for name in required_tokens) and qa.get("svg_embedded_raster_count") == 0,
        "required_visible_tokens": all(all(token in (FIGURE_DIR / name).read_text(encoding="utf-8") for token in tokens) for name, tokens in required_tokens.items()),
        "raster_resolution": all(value >= 2950 for value in qa.get("tiff_min_dimension_px", {}).values()) and all(value >= 1450 for value in qa.get("png_min_dimension_px", {}).values()),
        "image_content_and_margins": all(0.015 <= value <= 0.55 for value in qa.get("nonwhite_fraction", {}).values()) and all(value <= 0.03 for value in qa.get("edge_ink_fraction", {}).values()),
        "manual_visual_qa": qa.get("manual_visual_qa") == "PASS",
    }
    _require(all(checks.values()), f"T7.1.5 bundle verification failed: {[name for name, passed in checks.items() if not passed]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-bundle", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_bundle:
        checks = verify_bundle()
        print(json.dumps({"bundle": _relative(MANIFEST), "checks": checks}, ensure_ascii=False, indent=2))
        return 0
    if args.verify:
        print(json.dumps(verify(path=args.report), ensure_ascii=False, indent=2))
        return 0
    report = build_report(args.source_data, args.markdown)
    _atomic_json(args.report, report)
    print(json.dumps(verify(report, args.report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
