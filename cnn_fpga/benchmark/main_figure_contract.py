"""T7.1.2 evidence-bounded contracts for manuscript Figures 1 and 2."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark import claim_evidence_boundary_matrix as claim_matrix
from cnn_fpga.benchmark import route_a_board_measurement_gate as board_gate
from cnn_fpga.benchmark import route_a_v5_final_evidence_gate as v5_gate


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.1.2"
SCHEMA_VERSION = "t7.1.2-main-figure-contract-v1"
VERDICT = "PASS_MAIN_FIGURES_1_2_RESTRICTED_PREBOARD_CONTRACT"

DEFAULT_REPORT = ROOT / "docs/t7_1_2_main_figure_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_1_2_main_figure_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/main_figures_1_2_contract.md"
FIGURE_DIR = ROOT / "docs/figures/t7_1_2_main_figures"
DEFAULT_MANIFEST = FIGURE_DIR / "figure_manifest.json"

SOURCES = {
    "claim_report": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json",
    "claim_source": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix_source_data.csv",
    "claim_code": ROOT / "cnn_fpga/benchmark/claim_evidence_boundary_matrix.py",
    "claim_contract_report": ROOT / "docs/t6_5_1_route_a_claim_contract.json",
    "claim_contract_source": ROOT / "docs/t6_5_1_route_a_claim_contract_source_data.csv",
    "claim_contract_code": ROOT / "cnn_fpga/benchmark/route_a_claim_contract.py",
    "policy_report": ROOT / "docs/t6_6_2_regime_aware_safe_policy.json",
    "policy_source": ROOT / "docs/t6_6_2_regime_aware_safe_policy_source_data.csv",
    "policy_code": ROOT / "cnn_fpga/runtime/regime_aware_safe_policy.py",
    "rtl_report": ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json",
    "rtl_source": ROOT / "docs/t6_7_3_route_a_integrated_rtl_source_data.csv",
    "rtl_code": ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv",
    "pr_report": ROOT / "docs/t6_9_1_route_a_hardware_pareto.json",
    "pr_source": ROOT / "docs/t6_9_1_route_a_hardware_pareto_source_data.csv",
    "pr_code": ROOT / "cnn_fpga/benchmark/route_a_hardware_pareto.py",
    "board_report": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "board_code": ROOT / "cnn_fpga/benchmark/route_a_board_measurement_gate.py",
    "v5_report": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json",
    "v5_source": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate_source_data.csv",
    "v5_code": ROOT / "cnn_fpga/benchmark/route_a_v5_final_evidence_gate.py",
    "implementation": Path(__file__).resolve(),
}

FIGURE_OUTPUTS = (
    "figure1_contract_system.svg", "figure1_contract_system.pdf",
    "figure1_contract_system.png", "figure1_contract_system.tiff",
    "figure2_safe_adaptation.svg", "figure2_safe_adaptation.pdf",
    "figure2_safe_adaptation.png", "figure2_safe_adaptation.tiff",
)


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
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _parent_verification() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    calls = {
        "claim_matrix": lambda: claim_matrix.verify_report(path=SOURCES["claim_report"]),
        "board_blocker": lambda: board_gate.verify_report(_load(SOURCES["board_report"])),
        "v5_final": lambda: v5_gate.validate_report(SOURCES["v5_report"]),
    }
    for key, call in calls.items():
        try:
            call()
            checks[key] = True
        except Exception:
            checks[key] = False
    return checks


def _element(
    figure: str,
    panel: str,
    element_id: str,
    kind: str,
    label: str,
    status: str,
    layer: str,
    source_ids: Sequence[str],
    selectors: Sequence[str],
    claim_ids: Sequence[str],
    value: Any = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "figure": figure, "panel": panel, "element_id": element_id, "kind": kind,
        "label": label, "status": status, "evidence_layer": layer,
        "source_ids": list(source_ids), "selectors": list(selectors), "claim_ids": list(claim_ids),
        "value": value, "unit": unit,
    }


def _build_elements() -> list[dict[str, Any]]:
    policy = _load(SOURCES["policy_report"])
    rtl = _load(SOURCES["rtl_report"])
    pr = _load(SOURCES["pr_report"])
    board = _load(SOURCES["board_report"])
    v5 = _load(SOURCES["v5_report"])
    selected = next(row for row in pr["profiles"] if row["profile_id"] == "route_a_core_no_student")
    elements = [
        _element("Figure 1", "a", "f1a_plant", "node", "GKP plant / frozen simulator", "CONTEXT_NOT_HARDWARE", "PROJECT_NATIVE_SIMULATION", ["claim_report"], ["claims.CONTRACT_SYSTEM_INTEGRATION"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 1", "a", "f1a_observed", "node", "Observed syndrome + health / integrity", "IMPLEMENTED", "PROJECT_NATIVE_SIMULATION", ["policy_report", "policy_code"], ["contract", "RouteACycleInput"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 1", "a", "f1a_fast", "node", "FPGA MAP-LUT + event / action fast path", "PREBOARD_IMPLEMENTED", "CXXRTL_PREBOARD", ["rtl_report", "rtl_code"], ["integer_contract", "aggregate_python"], ["CONTRACT_SYSTEM_INTEGRATION", "FPGA_DETERMINISTIC_ARCHITECTURE"]),
        _element("Figure 1", "a", "f1a_action", "node", "Frame / correction / reset request", "DIGITAL_ACTION_CONTRACT", "CXXRTL_PREBOARD", ["rtl_report"], ["aggregate_python.actions"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 1", "a", "f1a_host", "node", "Observed-only host estimator", "SOFTWARE_SLOW_LOOP", "PROJECT_NATIVE_SIMULATION", ["policy_report", "policy_code"], ["contract", "claim_boundary"], ["CNN_AND_HMM_ROLE"]),
        _element("Figure 1", "a", "f1a_policy", "node", "Regime-aware typed safety policy", "SOFTWARE_IMPLEMENTED", "PROJECT_NATIVE_SIMULATION", ["policy_report", "policy_code"], ["trace.mode_counts", "trace.reason_counts"], ["CONTRACT_SYSTEM_INTEGRATION", "TAIL_SAFETY_AND_IMPROVEMENT"]),
        _element("Figure 1", "a", "f1a_bank", "node", "Versioned A/B trusted bank", "PREBOARD_IMPLEMENTED", "FIXED_POINT_INTEGER_REFERENCE", ["policy_report", "rtl_report"], ["trace.commit_versions", "integer_contract.ewma_bank", "integer_contract.window_bank"], ["CONTRACT_SYSTEM_INTEGRATION", "FPGA_DETERMINISTIC_ARCHITECTURE"]),
        _element("Figure 1", "a", "f1a_lkg", "node", "Last-known-good rollback + hysteresis", "PREBOARD_IMPLEMENTED", "CXXRTL_PREBOARD", ["policy_report", "rtl_report"], ["trace.rollback_completed_cycles", "integer_contract.recovery_hysteresis"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 1", "a", "f1a_fast_edge", "edge", "solid: per-round fast path", "IMPLEMENTED", "CXXRTL_PREBOARD", ["rtl_report"], ["integer_contract"], ["FPGA_DETERMINISTIC_ARCHITECTURE"]),
        _element("Figure 1", "a", "f1a_slow_edge", "edge", "dashed: host parameter update", "SOFTWARE_ONLY", "PROJECT_NATIVE_SIMULATION", ["policy_report"], ["contract.parameter_update_period_cycles"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 1", "a", "f1a_learning_sidecar", "boundary", "CNN / teacher / student: optional ablation sidecar", "NOT_PRIMARY_NOT_FAST_ACTION", "PROJECT_NATIVE_SIMULATION", ["pr_report", "claim_report"], ["pareto_decision.student_profile_role", "claims.CNN_AND_HMM_ROLE"], ["CNN_AND_HMM_ROLE"]),
        _element("Figure 1", "b", "f1b_sim", "evidence", "Project-native simulation", "AVAILABLE_RESTRICTED", "PROJECT_NATIVE_SIMULATION", ["claim_report"], ["evidence_layer_ontology.PROJECT_NATIVE_SIMULATION"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 1", "b", "f1b_fixed", "evidence", "Fixed-point integer reference", "AVAILABLE_RESTRICTED", "FIXED_POINT_INTEGER_REFERENCE", ["rtl_report"], ["integer_contract"], ["FPGA_DETERMINISTIC_ARCHITECTURE"]),
        _element("Figure 1", "b", "f1b_cxxrtl", "evidence", "CXXRTL pre-board qualification", "AVAILABLE_RESTRICTED", "CXXRTL_PREBOARD", ["rtl_report"], ["aggregate_python.cycles", "aggregate_python.undefined_actions", "aggregate_python.silent_overflow"], ["FPGA_DETERMINISTIC_ARCHITECTURE"], value=rtl["aggregate_python"]["cycles"], unit="cycles"),
        _element("Figure 1", "b", "f1b_pr", "evidence", "Three-seed place-and-route estimate", "AVAILABLE_ESTIMATE", "POST_ROUTE_ESTIMATE", ["pr_report", "pr_source"], ["profiles.route_a_core_no_student.summary"], ["FPGA_DETERMINISTIC_ARCHITECTURE"], value=selected["summary"]["fmax_mhz"]["minimum"], unit="MHz minimum Fmax estimate"),
        _element("Figure 1", "b", "f1b_board", "evidence", "Physical-board measurement", "BLOCKED_ALL_FIELDS_NULL", "BOARD_MEASURED", ["board_report", "board_code"], ["measured_results", "claim_boundary"], ["BOARD_MEASURED_CORRECTNESS_LATENCY", "FPGA_SPEED_ADVANTAGE"], value=sum(value is None for value in board["measured_results"].values()), unit="null fields"),
        _element("Figure 1", "c", "f1c_pipeline", "timing", "source → action", "CLOCK_MODEL_NOT_BOARD", "CXXRTL_PREBOARD", ["pr_report", "rtl_report"], ["source_to_action_latency_model"], ["FPGA_DETERMINISTIC_ARCHITECTURE"], value=selected["source_to_action_latency_model"]["cycles"], unit="cycles"),
        _element("Figure 1", "c", "f1c_ii", "timing", "initiation interval", "CLOCK_MODEL_NOT_BOARD", "CXXRTL_PREBOARD", ["pr_report"], ["source_to_action_latency_model.initiation_interval_cycles"], ["FPGA_DETERMINISTIC_ARCHITECTURE"], value=selected["source_to_action_latency_model"]["initiation_interval_cycles"], unit="cycles"),
        _element("Figure 1", "c", "f1c_update", "timing", "host update cadence", "SOFTWARE_CONTRACT", "PROJECT_NATIVE_SIMULATION", ["policy_report"], ["contract.parameter_update_period_cycles"], ["CONTRACT_SYSTEM_INTEGRATION"], value=policy["contract"]["parameter_update_period_cycles"], unit="cycles"),
        _element("Figure 1", "c", "f1c_board_latency", "timing", "board source-to-action latency", "NULL_BLOCKED", "BOARD_MEASURED", ["board_report"], ["measured_results.source_to_action_p50_ns"], ["BOARD_MEASURED_CORRECTNESS_LATENCY"], value=None, unit="ns"),
        _element("Figure 2", "a", "f2a_syndrome", "input", "q/p syndrome codes", "OBSERVED_ONLY", "FIXED_POINT_INTEGER_REFERENCE", ["rtl_report", "rtl_code"], ["integer_contract", "input"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "a", "f2a_health", "input", "health / leakage / integrity flags", "OBSERVED_ONLY", "CXXRTL_PREBOARD", ["rtl_report", "policy_report"], ["aggregate_python.reasons", "RouteACycleInput"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "a", "f2a_version", "input", "version / CRC / age / ack", "OBSERVED_ONLY", "CXXRTL_PREBOARD", ["rtl_report", "policy_report"], ["aggregate_python.core_fault_bits", "contract.max_parameter_age_cycles"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "b", "f2b_smooth", "regime", "normal / smooth", "IMPLEMENTED_POLICY_BRANCH", "PROJECT_NATIVE_SIMULATION", ["policy_report"], ["trace.reason_counts.normal_or_smooth_posterior"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "b", "f2b_tail", "regime", "calibration shift / burst", "IMPLEMENTED_POLICY_BRANCH", "PROJECT_NATIVE_SIMULATION", ["policy_report"], ["trace.reason_counts.calibration_shift_or_burst_posterior"], ["TAIL_SAFETY_AND_IMPROVEMENT"]),
        _element("Figure 2", "b", "f2b_leakage", "regime", "leakage", "IMPLEMENTED_POLICY_BRANCH", "PROJECT_NATIVE_SIMULATION", ["policy_report", "rtl_report"], ["trace.reason_counts.leakage_observed", "aggregate_python.leakage_entries"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "b", "f2b_integrity", "regime", "uncertainty / CRC / version", "IMPLEMENTED_POLICY_BRANCH", "CXXRTL_PREBOARD", ["policy_report", "rtl_report"], ["trace.reason_counts.posterior_uncertain", "aggregate_python.integrity_entries"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "c", "f2c_update", "action", "stage eligible adaptive image", "TYPED_ACTION", "PROJECT_NATIVE_SIMULATION", ["policy_report"], ["trace.candidate_rows"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "c", "f2c_trusted", "action", "freeze update; select trusted bank", "TYPED_ACTION", "CXXRTL_PREBOARD", ["rtl_report"], ["aggregate_python.actions.tail_ewma"], ["TAIL_SAFETY_AND_IMPROVEMENT"]),
        _element("Figure 2", "c", "f2c_reset", "action", "leakage reset / frame hold", "TYPED_ACTION", "CXXRTL_PREBOARD", ["rtl_report"], ["aggregate_python.actions.leakage_reset"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "c", "f2c_rollback", "action", "rollback to last-known-good", "TYPED_ACTION", "CXXRTL_PREBOARD", ["rtl_report", "policy_report"], ["aggregate_python.actions.integrity_rollback", "trace.rollback_completed_cycles"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "d", "f2d_candidate", "transaction", "candidate image: CRC/SHA/version", "ATOMIC_PRECONDITION", "FIXED_POINT_INTEGER_REFERENCE", ["policy_report", "rtl_report"], ["trace.candidate_rows", "integer_contract"], ["CONTRACT_SYSTEM_INTEGRATION"]),
        _element("Figure 2", "d", "f2d_inactive", "transaction", "write inactive A/B bank", "ATOMIC_PRECONDITION", "CXXRTL_PREBOARD", ["rtl_report"], ["aggregate_python.selected_banks"], ["FPGA_DETERMINISTIC_ARCHITECTURE"]),
        _element("Figure 2", "d", "f2d_commit", "transaction", "safe-boundary atomic commit", "ATOMIC_PRECONDITION", "CXXRTL_PREBOARD", ["rtl_report", "policy_report"], ["aggregate_python.commit_acks", "trace.commit_versions"], ["FPGA_DETERMINISTIC_ARCHITECTURE"]),
        _element("Figure 2", "d", "f2d_recover", "transaction", "LKG republish + recovery hysteresis", "ATOMIC_PRECONDITION", "CXXRTL_PREBOARD", ["rtl_report", "policy_report"], ["integer_contract.recovery_hysteresis", "trace.rollback_completed_cycles"], ["CONTRACT_SYSTEM_INTEGRATION"], value=rtl["integer_contract"]["recovery_hysteresis"], unit="windows"),
        _element("Figure 2", "e", "f2e_v5", "boundary", "IMM / BOCPD; posterior-mixture MAP; V5 risk compiler", "NOT_RUN_DROPPED", "PROJECT_NATIVE_SIMULATION", ["v5_report", "v5_source", "v5_code"], ["execution_path", "dropped_tasks", "claim_registry"], ["V5-POSTERIOR-MIXTURE-ACTION", "V5-UNTOUCHED-FORMAL"], value=len(v5["dropped_tasks"]), unit="dropped tasks"),
        _element("Figure 2", "e", "f2e_v5_rtl", "boundary", "V5 quantized / formal / CXXRTL / P&R", "NOT_RUN_DROPPED", "CXXRTL_PREBOARD", ["v5_report"], ["claim_registry"], ["V5-QUANTIZED-RETENTION", "V5-LONG-CXXRTL", "V5-FORMAL-ATOMIC-SAFETY", "V5-MULTISEED-PR"]),
        _element("Figure 2", "e", "f2e_board", "boundary", "board latency / jitter / power", "BLOCKED_NULL", "BOARD_MEASURED", ["board_report", "v5_report"], ["measured_results", "measured_hardware_claim"], ["V5-MEASURED-HARDWARE", "BOARD_MEASURED_CORRECTNESS_LATENCY"]),
    ]
    return elements


def _write_source_data(elements: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    fields = ["figure", "panel", "element_id", "kind", "label", "status", "evidence_layer", "value_json", "unit", "source_ids_json", "source_hashes_json", "selectors_json", "claim_ids_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in elements:
            writer.writerow({
                "figure": row["figure"], "panel": row["panel"], "element_id": row["element_id"],
                "kind": row["kind"], "label": row["label"], "status": row["status"],
                "evidence_layer": row["evidence_layer"], "value_json": json.dumps(row["value"], ensure_ascii=False),
                "unit": row["unit"] or "", "source_ids_json": json.dumps(row["source_ids"], separators=(",", ":")),
                "source_hashes_json": json.dumps({key: artifacts[key]["sha256"] for key in row["source_ids"]}, sort_keys=True, separators=(",", ":")),
                "selectors_json": json.dumps(row["selectors"], ensure_ascii=False, separators=(",", ":")),
                "claim_ids_json": json.dumps(row["claim_ids"], ensure_ascii=False, separators=(",", ":")),
            })
    temporary.replace(path)


def _render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "# T7.1.2 主图 1--2 冻结合同", "",
        "## Figure contract", "",
        "- Backend：Python / matplotlib only。",
        "- Archetype：两图均为 schematic-led composite；183 mm 双栏，白底、可编辑 SVG/PDF、600-dpi LZW-TIFF。",
        "- Fig.1 核心结论：受支持的贡献是 observed-only host slow loop 通过 versioned trusted bank 接入 6-cycle/II=1 FPGA fast path 的预板合同；P&R 是 estimate，board measurement 仍为空。",
        "- Fig.2 核心结论：安全由 typed event/action、freeze/switch/reset/LKG rollback 和 hysteresis 实现；IMM/BOCPD、posterior-mixture MAP 与 V5 risk compiler 明确标为 not run/dropped。", "",
        "## Panel map", "",
        "- Fig.1a：双回路 hero schematic；实线为逐轮 fast path，虚线为 host parameter update。",
        "- Fig.1b：simulation → fixed-point → CXXRTL → P&R estimate → board-null 证据层。",
        "- Fig.1c：6-cycle source-to-action、II=1、4000-cycle host cadence 与 board latency null。",
        "- Fig.2a--c：observed-only 输入、四类证据分支与四类 typed action。",
        "- Fig.2d：candidate → inactive bank → safe commit → LKG/hysteresis 事务链。",
        "- Fig.2e：Dropped V5 与 blocked board 明示，不作为淡化脚注。", "",
        "## Reviewer-risk checks", "",
        "1. CNN/teacher/student 仅为 optional ablation sidecar，不驱动 fast action；HMM/posterior inference 位于软件慢回路。",
        "2. `POST_ROUTE_ESTIMATE` 与 `BOARD_MEASURED` 颜色和标签不同；42 个 measured 字段不填零。",
        "3. V5 early-stop 模块不使用已实现配色，也不连入 production arrows。",
        "4. 所有节点/边均在 Source Data 中绑定 report/Source Data/code SHA-256 与 selector。", "",
        "## Figure legends", "",
        "**Fig. 1 | Evidence-bounded dual-loop Route-A contract.** a, Observed syndrome and integrity inputs feed a deterministic MAP-LUT/event fast path, while a software slow loop stages versioned parameter images through trusted A/B banks and last-known-good recovery. Solid and dashed arrows denote per-round and update-cadence paths, respectively. CNN/teacher/student modules remain optional ablations. b, Available evidence progresses from project-native simulation through fixed-point and CXXRTL qualification to a three-seed post-route estimate; physical-board measurement is blocked and all 42 measured fields remain null. c, The pre-board timing contract is six cycles with initiation interval one, whereas host updates occur every 4000 cycles. These values are not board-measured latency. Source data are provided as a Source Data file.", "",
        "**Fig. 2 | Typed fail-closed adaptation and atomic parameter control.** a--c, Observed syndrome, health, integrity, version and age fields select typed normal/smooth, tail, leakage and integrity branches and their corresponding stage, trusted-bank, reset or last-known-good actions. d, Candidate images cross CRC/SHA/version checks before inactive-bank write and safe-boundary atomic commit; recovery requires hysteresis. e, IMM/BOCPD, posterior-mixture MAP, the V5 risk compiler and V5 implementation results were not run after the preregistered early stop, while board latency, jitter and power remain blocked. Source data are provided as a Source Data file.", "",
        f"Machine verdict: `{report['verdict']}`.", "",
    ])


def _csv_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["element_id"] for row in csv.DictReader(handle)]


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    elements = {row["element_id"]: row for row in report["elements"]}
    artifacts = report["artifact_registry"]
    current_parents = _parent_verification() if check_live_files else report["parent_verification"]
    source_path = ROOT / report["source_data"]["path"]
    markdown_path = ROOT / report["markdown"]["path"]
    implemented_labels = " ".join(row["label"].lower() for row in elements.values() if row["status"] not in {"NOT_RUN_DROPPED", "BLOCKED_NULL", "BLOCKED_ALL_FIELDS_NULL", "NULL_BLOCKED"})
    dropped_labels = " ".join(row["label"].lower() for row in elements.values() if row["status"] == "NOT_RUN_DROPPED")
    evidence_ids = {source for row in elements.values() for source in row["source_ids"]}
    claims = _load(SOURCES["claim_report"])
    valid_claim_ids = {row["claim_id"] for row in claims["claims"]}
    return {
        "G01_parent_claim_board_and_v5_verifiers_pass_live": all(current_parents.values()) and report["parent_verification"] == current_parents,
        "G02_all_sources_are_live_and_every_element_is_hash_traceable": set(artifacts) == set(SOURCES) and evidence_ids <= set(artifacts) and all(len(row["sha256"]) == 64 for row in artifacts.values()) and (not check_live_files or all(_live(row) for row in artifacts.values())),
        "G03_two_figure_contracts_have_conclusion_archetype_size_and_panel_map": set(report["figures"]) == {"Figure 1", "Figure 2"} and all(row["core_conclusion"] and row["archetype"] == "schematic-led composite" and row["width_mm"] == 183 and row["height_mm"] == expected_height and row["panel_map"] for row, expected_height in ((report["figures"]["Figure 1"], 127), (report["figures"]["Figure 2"], 137))),
        "G04_elements_are_unique_complete_and_source_bound": len(elements) == len(report["elements"]) == 38 and all(row["figure"] in report["figures"] and row["panel"] in report["figures"][row["figure"]]["panel_map"] and row["source_ids"] and row["selectors"] and row["claim_ids"] for row in elements.values()),
        "G05_figure1_separates_fast_slow_learning_and_physical_layers": all(key in elements for key in ("f1a_fast", "f1a_host", "f1a_fast_edge", "f1a_slow_edge", "f1a_learning_sidecar", "f1b_board")) and elements["f1a_fast"]["evidence_layer"] == "CXXRTL_PREBOARD" and elements["f1a_host"]["status"] == "SOFTWARE_SLOW_LOOP" and elements["f1a_learning_sidecar"]["status"] == "NOT_PRIMARY_NOT_FAST_ACTION",
        "G06_six_cycle_ii1_and_board_null_are_exact": elements["f1c_pipeline"]["value"] == 6 and elements["f1c_ii"]["value"] == 1 and elements["f1b_board"]["value"] == 42 and elements["f1c_board_latency"]["value"] is None,
        "G07_postroute_and_board_measurement_are_never_merged": elements["f1b_pr"]["evidence_layer"] == "POST_ROUTE_ESTIMATE" and elements["f1b_board"]["evidence_layer"] == "BOARD_MEASURED" and elements["f1b_pr"]["status"] == "AVAILABLE_ESTIMATE" and elements["f1b_board"]["status"] == "BLOCKED_ALL_FIELDS_NULL",
        "G08_figure2_has_four_observed_inputs_regimes_and_typed_actions": all(key in elements for key in ("f2a_syndrome", "f2a_health", "f2a_version", "f2b_smooth", "f2b_tail", "f2b_leakage", "f2b_integrity", "f2c_update", "f2c_trusted", "f2c_reset", "f2c_rollback")) and all(elements[key]["status"] in {"OBSERVED_ONLY", "IMPLEMENTED_POLICY_BRANCH", "TYPED_ACTION"} for key in ("f2a_syndrome", "f2a_health", "f2a_version", "f2b_smooth", "f2b_tail", "f2b_leakage", "f2b_integrity", "f2c_update", "f2c_trusted", "f2c_reset", "f2c_rollback")),
        "G09_atomic_bank_lkg_version_and_hysteresis_are_explicit": all(key in elements for key in ("f2d_candidate", "f2d_inactive", "f2d_commit", "f2d_recover")) and elements["f2d_recover"]["value"] == 8 and all(row["status"] == "ATOMIC_PRECONDITION" for key, row in elements.items() if key.startswith("f2d_")),
        "G10_v5_imm_bocpd_mixture_risk_and_implementation_are_dropped_only": all(token in dropped_labels for token in ("imm", "bocpd", "posterior-mixture", "v5 risk", "v5 quantized", "formal", "cxxrtl", "p&r")) and not any(token in implemented_labels for token in ("imm", "bocpd", "posterior-mixture", "v5 risk compiler")),
        "G11_no_board_hmm_or_cnn_fastpath_promotion": "board-measured" not in implemented_labels and elements["f1a_learning_sidecar"]["status"] == "NOT_PRIMARY_NOT_FAST_ACTION" and report["forbidden_promotions"] == ["HMM or CNN in RTL", "V5 module implemented", "post-route estimate as board measurement", "measured speed or power advantage"],
        "G12_every_claim_reference_exists_in_t7_1_1_matrix": all(set(row["claim_ids"]) <= valid_claim_ids for row in elements.values()),
        "G13_export_contract_is_python_only_editable_and_multiformat": report["export_contract"] == {"backend": "Python/matplotlib only", "width_mm": 183, "svg_text": "editable", "pdf_fonttype": 42, "tiff_dpi": 600, "png_dpi": 300, "outputs": list(FIGURE_OUTPUTS)},
        "G14_source_data_is_lossless_one_row_per_element": source_path.is_file() and report["source_data"]["rows"] == len(elements) and set(_csv_ids(source_path)) == set(elements) and len(_csv_ids(source_path)) == len(elements) and (not check_live_files or _live(report["source_data"])),
        "G15_human_contract_is_live_and_contains_legends_and_boundaries": markdown_path.is_file() and all(token in markdown_path.read_text(encoding="utf-8") for token in ("Fig. 1 |", "Fig. 2 |", "board measurement 仍为空", "not run/dropped", "Source data are provided")) and (not check_live_files or _live(report["markdown"])),
        "G16_one_substantive_mutation_per_gate_fails_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 16 and len(report["semantic_mutation_audit"]["cases"]) == 16,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def element(value: dict[str, Any], element_id: str) -> dict[str, Any]:
        return next(row for row in value["elements"] if row["element_id"] == element_id)

    def attempt(name: str, target: str, change: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
        change(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[target]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": target, "rejected": rejected})

    attempt("parent_failure", "G01_parent_claim_board_and_v5_verifiers_pass_live", lambda x: x["parent_verification"].update(v5_final=False))
    attempt("forge_source_hash", "G02_all_sources_are_live_and_every_element_is_hash_traceable", lambda x: x["artifact_registry"]["policy_report"].update(sha256="0"))
    attempt("drop_conclusion", "G03_two_figure_contracts_have_conclusion_archetype_size_and_panel_map", lambda x: x["figures"]["Figure 1"].update(core_conclusion=""))
    attempt("duplicate_element", "G04_elements_are_unique_complete_and_source_bound", lambda x: x["elements"][-1].update(element_id=x["elements"][0]["element_id"]))
    attempt("cnn_drives_action", "G05_figure1_separates_fast_slow_learning_and_physical_layers", lambda x: element(x, "f1a_learning_sidecar").update(status="PRIMARY_FAST_ACTION"))
    attempt("shorten_pipeline", "G06_six_cycle_ii1_and_board_null_are_exact", lambda x: element(x, "f1c_pipeline").update(value=1))
    attempt("promote_pr_to_board", "G07_postroute_and_board_measurement_are_never_merged", lambda x: element(x, "f1b_pr").update(evidence_layer="BOARD_MEASURED"))
    attempt("drop_integrity_action", "G08_figure2_has_four_observed_inputs_regimes_and_typed_actions", lambda x: element(x, "f2c_rollback").update(status="UNDEFINED"))
    attempt("erase_hysteresis", "G09_atomic_bank_lkg_version_and_hysteresis_are_explicit", lambda x: element(x, "f2d_recover").update(value=0))
    attempt("promote_v5", "G10_v5_imm_bocpd_mixture_risk_and_implementation_are_dropped_only", lambda x: element(x, "f2e_v5").update(status="IMPLEMENTED_POLICY_BRANCH"))
    attempt("remove_forbidden_boundary", "G11_no_board_hmm_or_cnn_fastpath_promotion", lambda x: x.update(forbidden_promotions=[]))
    attempt("invent_claim", "G12_every_claim_reference_exists_in_t7_1_1_matrix", lambda x: element(x, "f1a_fast")["claim_ids"].append("SOTA"))
    attempt("switch_backend", "G13_export_contract_is_python_only_editable_and_multiformat", lambda x: x["export_contract"].update(backend="R"))
    attempt("forge_source_rows", "G14_source_data_is_lossless_one_row_per_element", lambda x: x["source_data"].update(rows=0))
    attempt("disconnect_markdown", "G15_human_contract_is_live_and_contains_legends_and_boundaries", lambda x: x["markdown"].update(path="docs/not_found.md"))
    attempt("forge_mutation_count", "G16_one_substantive_mutation_per_gate_fails_closed", lambda x: x.update(semantic_mutation_audit={"count": 16, "detected": 15, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report(source_data: Path = DEFAULT_SOURCE_DATA, markdown: Path = DEFAULT_MARKDOWN) -> dict[str, Any]:
    artifacts = {key: _binding(path) for key, path in SOURCES.items()}
    elements = _build_elements()
    _write_source_data(elements, artifacts, source_data)
    report: dict[str, Any] = {
        "task_id": TASK_ID, "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figures": {
            "Figure 1": {
                "core_conclusion": "The supported contribution is an observed-only host slow loop coupled through versioned trusted banks to a deterministic six-cycle, II=1 pre-board FPGA fast path; physical-board performance remains unavailable.",
                "archetype": "schematic-led composite", "width_mm": 183, "height_mm": 127,
                "panel_map": {"a": "dual-loop hero schematic", "b": "evidence ladder", "c": "timing and ownership"},
            },
            "Figure 2": {
                "core_conclusion": "Typed observed evidence drives fail-closed stage, trusted-bank, reset and last-known-good actions with atomic commit and hysteresis; V5 mixture modules remain explicitly not run.",
                "archetype": "schematic-led composite", "width_mm": 183, "height_mm": 137,
                "panel_map": {"a": "observed-only inputs", "b": "regime evidence", "c": "typed actions", "d": "atomic A/B and LKG transaction", "e": "dropped and blocked boundary"},
            },
        },
        "artifact_registry": artifacts,
        "parent_verification": _parent_verification(),
        "elements": elements,
        "forbidden_promotions": ["HMM or CNN in RTL", "V5 module implemented", "post-route estimate as board measurement", "measured speed or power advantage"],
        "export_contract": {"backend": "Python/matplotlib only", "width_mm": 183, "svg_text": "editable", "pdf_fonttype": 42, "tiff_dpi": 600, "png_dpi": 300, "outputs": list(FIGURE_OUTPUTS)},
        "source_data": {**_binding(source_data), "rows": len(elements)},
        "markdown": {"path": _relative(markdown), "sha256": "", "bytes": 0},
    }
    report["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
    report["verdict"] = VERDICT
    _atomic_text(_render_markdown(report), markdown)
    report["markdown"] = _binding(markdown)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": [key for key, value in report["gates"].items() if not value]}
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_MAIN_FIGURE_CONTRACT"
    report["analysis_sha256"] = _canonical_sha256({key: report[key] for key in ("figures", "artifact_registry", "parent_verification", "elements", "forbidden_promotions", "export_contract", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")})
    return report


def verify_report(report: Mapping[str, Any] | None = None, path: Path = DEFAULT_REPORT) -> dict[str, bool]:
    value = dict(report) if report is not None else _load(path)
    gates = evaluate_gates(value)
    expected_hash = _canonical_sha256({key: value[key] for key in ("figures", "artifact_registry", "parent_verification", "elements", "forbidden_promotions", "export_contract", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")})
    checks = {
        "identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION,
        "gates": value.get("gates") == gates and all(gates.values()),
        "verdict": value.get("verdict") == VERDICT,
        "analysis_hash": value.get("analysis_sha256") == expected_hash,
    }
    if not all(checks.values()):
        raise ValueError(f"T7.1.2 contract verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def verify_bundle(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, bool]:
    manifest = _load(manifest_path)
    outputs = manifest.get("outputs", {})
    checks = {
        "identity": manifest.get("task_id") == TASK_ID and manifest.get("backend") == "Python/matplotlib only",
        "contract_live": manifest.get("contract") == _binding(DEFAULT_REPORT),
        "source_data_live": manifest.get("source_data") == _binding(DEFAULT_SOURCE_DATA),
        "outputs_exact": set(outputs) == set(FIGURE_OUTPUTS),
        "outputs_live": set(outputs) == set(FIGURE_OUTPUTS) and all(_live(binding) for binding in outputs.values()),
        "editable_svg": manifest.get("qa", {}).get("svg_text_nodes", 0) >= 50 and manifest.get("qa", {}).get("svg_path_text_promotion") is False,
        "raster_dimensions": all(value >= 3000 for value in manifest.get("qa", {}).get("tiff_min_dimension_px", {}).values()),
        "visual_contract": manifest.get("qa", {}).get("backend_exclusive") is True and manifest.get("qa", {}).get("manual_visual_qa") == "PASS",
    }
    if not all(checks.values()):
        raise ValueError(f"T7.1.2 bundle verification failed: {[key for key, passed in checks.items() if not passed]}")
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
    print(json.dumps({"output": _relative(args.report), "elements": len(report["elements"]), "gates": report["gate_summary"], "verdict": report["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
