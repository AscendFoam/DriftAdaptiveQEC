"""Build the T7.1.1 fail-closed claim--evidence--boundary matrix."""

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

from cnn_fpga.benchmark import route_a_board_measurement_gate as board_gate
from cnn_fpga.benchmark import route_a_claim_matrix as claim_matrix_gate
from cnn_fpga.benchmark import route_a_final_evidence_gate as v4_gate
from cnn_fpga.benchmark import route_a_v5_final_evidence_gate as v5_gate
from cnn_fpga.benchmark import secondary_evidence_integrity_gate as phase6c_gate


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.1.1"
SCHEMA_VERSION = "t7.1.1-claim-evidence-boundary-matrix-v1"
VERDICT = "PASS_RESTRICTED_PREBOARD_CLAIM_BOUNDARY_MATRIX"

DEFAULT_REPORT = ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/claim_evidence_boundary_matrix.md"
BOARD = ROOT / "docs/new_task_board.md"

CLAIM_MATRIX = ROOT / "docs/t6_8_7_route_a_claim_matrix.json"
V4 = ROOT / "docs/t6_9_3_route_a_final_evidence_gate.json"
V5 = ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json"
PHASE6C = ROOT / "docs/t6_19_3_secondary_evidence_integrity.json"
BOARD_BLOCKER = ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json"

ARTIFACT_PATHS = {
    "claim_matrix_report": CLAIM_MATRIX,
    "claim_matrix_source": ROOT / "docs/t6_8_7_route_a_claim_matrix_source_data.csv",
    "claim_matrix_code": ROOT / "cnn_fpga/benchmark/route_a_claim_matrix.py",
    "v4_report": V4,
    "v4_source": ROOT / "docs/t6_9_3_route_a_final_evidence_gate_source_data.csv",
    "v4_code": ROOT / "cnn_fpga/benchmark/route_a_final_evidence_gate.py",
    "v5_report": V5,
    "v5_source": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate_source_data.csv",
    "v5_code": ROOT / "cnn_fpga/benchmark/route_a_v5_final_evidence_gate.py",
    "phase6c_report": PHASE6C,
    "phase6c_source": ROOT / "docs/t6_19_3_secondary_evidence_integrity_source_data.csv",
    "phase6c_code": ROOT / "cnn_fpga/benchmark/secondary_evidence_integrity_gate.py",
    "single_cpd_report": ROOT / "docs/t6_17_1_single_mode_cpd_equivalence.json",
    "single_cpd_source": ROOT / "docs/t6_17_1_single_mode_cpd_equivalence_source_data.csv",
    "single_cpd_code": ROOT / "cnn_fpga/benchmark/single_mode_cpd_equivalence.py",
    "surface_cnot_report": ROOT / "docs/t6_17_2_noh_cnot_ci_ml_reproduction.json",
    "surface_cnot_source": ROOT / "docs/t6_17_2_noh_cnot_ci_ml_reproduction_source_data.csv",
    "surface_cnot_code": ROOT / "cnn_fpga/benchmark/surface_gkp_cnot_reproduction.py",
    "learned_report": ROOT / "docs/t6_17_3_learned_model_eligibility_replay.json",
    "learned_source": ROOT / "docs/t6_17_3_learned_model_eligibility_replay_source_data.csv",
    "learned_code": ROOT / "cnn_fpga/benchmark/learned_model_eligibility_replay.py",
    "gqf_exact_report": ROOT / "docs/t6_8_4_gqf_paper_exact_reproduction.json",
    "gqf_exact_source": ROOT / "docs/t6_8_4_gqf_reproduction_source_data.csv",
    "gqf_exact_code": ROOT / "cnn_fpga/benchmark/gqf_paper_exact_reproduction.py",
    "gqf_gate_report": ROOT / "docs/t6_8_5_gqf_route_a_matched_comparison_gate.json",
    "gqf_gate_source": ROOT / "docs/t6_8_5_gqf_route_a_matched_comparison_gate_source_data.csv",
    "gqf_gate_code": ROOT / "cnn_fpga/benchmark/gqf_route_a_matched_comparison_gate.py",
    "aqec_report": ROOT / "docs/t6_18_1_aqec_common_wallclock_replay.json",
    "aqec_source": ROOT / "docs/t6_18_1_aqec_common_wallclock_source_data.csv",
    "aqec_code": ROOT / "cnn_fpga/benchmark/aqec_secondary_wallclock_replay.py",
    "official_cpd_report": ROOT / "docs/t6_18_2_official_structured_cpd_reproduction.json",
    "official_cpd_source": ROOT / "docs/t6_18_2_official_structured_cpd_source_data.csv",
    "official_cpd_code": ROOT / "cnn_fpga/benchmark/official_structured_cpd_reproduction.py",
    "multimode_report": ROOT / "docs/t6_18_3_multimode_posterior_weighted_cpd.json",
    "multimode_source": ROOT / "docs/t6_18_3_multimode_posterior_weighted_cpd_source_data.csv",
    "multimode_code": ROOT / "cnn_fpga/benchmark/multimode_posterior_weighted_cpd.py",
    "preboard_report": ROOT / "docs/t6_19_1_project_preboard_profiles.json",
    "preboard_source": ROOT / "docs/t6_19_1_project_preboard_profiles_source_data.csv",
    "preboard_code": ROOT / "cnn_fpga/benchmark/phase6c_preboard_profiles.py",
    "external_fpga_report": ROOT / "docs/t6_19_2_external_fpga_normalization.json",
    "external_fpga_source": ROOT / "docs/t6_19_2_external_fpga_normalization_source_data.csv",
    "external_fpga_code": ROOT / "cnn_fpga/benchmark/external_fpga_decoder_refresh.py",
    "board_report": BOARD_BLOCKER,
    "board_code": ROOT / "cnn_fpga/benchmark/route_a_board_measurement_gate.py",
    "implementation": Path(__file__).resolve(),
}

PLACEMENTS = ("title", "abstract", "results", "conclusion", "supplement", "related_work")
PUBLICATION_STATES = {
    "ALLOWED_RESTRICTED", "RESULTS_ONLY", "RELATED_WORK_ONLY", "MANDATORY_NEGATIVE",
    "PROHIBITED_POSITIVE", "BLOCKED", "META_BOUNDARY",
}
EVIDENCE_GRADES = {
    "PROJECT_NATIVE_MATCHED", "PROJECT_NATIVE_PREBOARD", "OFFICIAL_CODE_REPRODUCTION",
    "LITERATURE_ONLY", "NEGATIVE", "BLOCKED", "META",
}
EVIDENCE_LAYERS = {
    "LITERATURE_ONLY", "OFFICIAL_CODE_REPRODUCTION", "PROJECT_NATIVE_SIMULATION",
    "FIXED_POINT_INTEGER_REFERENCE", "CXXRTL_PREBOARD", "POST_ROUTE_ESTIMATE",
    "BOARD_MEASURED",
}
V4_IDS = {
    "CONTRACT_SYSTEM_INTEGRATION", "SMOOTH_LOCKED_EWMA_ADVANTAGE", "STATIC_GKP_SUPERIORITY",
    "STATIC_K4_HARD_ACTION_EQUIVALENCE", "TAIL_SAFETY_AND_IMPROVEMENT",
    "GENERAL_DRIFT_EXTERNAL_COMPARISON", "PUVIANI_NMF_SURPASS",
    "FPGA_DETERMINISTIC_ARCHITECTURE", "BOARD_MEASURED_CORRECTNESS_LATENCY",
    "FPGA_SPEED_ADVANTAGE", "CNN_AND_HMM_ROLE",
}
V5_IDS = {
    "V5-ALG-LER-10PCT", "V5-TAIL-CALIBRATION-TELEGRAPH", "V5-POSTERIOR-MIXTURE-ACTION",
    "V5-UNTOUCHED-FORMAL", "V5-QUANTIZED-RETENTION", "V5-LONG-CXXRTL",
    "V5-FORMAL-ATOMIC-SAFETY", "V5-MULTISEED-PR", "V5-MEASURED-HARDWARE",
    "PHASE6C-READONLY-AUX",
}
PHASE6C_IDS = {
    "P6C-SINGLE-CI-CPD-EQUIVALENCE", "P6C-SURFACE-NOH-CI-ML",
    "P6C-STRUCTURED-OFFICIAL-CPD", "P6C-MULTIMODE-POSTERIOR-WEIGHTED",
    "P6C-LEARNED-ELIGIBILITY", "P6C-NMF-GQF", "P6C-AQEC-WALLCLOCK",
    "P6C-FPGA-NORMALIZATION",
}
LANES = {
    "single_mode_decoder", "surface_gkp_gate_outer_code", "multimode_structured_lattice_cpd",
    "controller_rl_nmf", "aqec_wallclock", "fpga_implementation",
}

TERMINOLOGY_LEDGER = {
    "primary_system": "Route-A contract system",
    "static_baseline": "static joint MAP",
    "multimode_extension": "observed-only posterior-predictive weighted CPD",
    "latency_metric": "source-to-action latency",
    "hardware_estimate": "post-route estimate",
    "physical_evidence": "board measurement",
    "auxiliary_scope": "secondary evidence lane",
    "oracle": "hidden-state oracle",
    "learning_component": "legacy CNN residual",
}


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


def _task_statuses(text: str) -> dict[str, str]:
    rows = re.findall(r"^\| (T[^| ]+) \| ([^|]+) \|", text, flags=re.MULTILINE)
    result: dict[str, str] = {}
    for task, status in rows:
        result.setdefault(task.strip(), status.strip())
    return result


def _board_binding(text: str) -> dict[str, Any]:
    statuses = _task_statuses(text)
    selected = {
        "T6.14.3": statuses["T6.14.3"],
        "T6.15.5": statuses["T6.15.5"],
        "T6.19.3": statuses["T6.19.3"],
        "T7.1.1": "ACTIVE_OR_DONE" if statuses["T7.1.1"] in {"In Progress", "Done"} else statuses["T7.1.1"],
    }
    return {"path": _relative(BOARD), "statuses": selected, "canonical_sha256": _canonical_sha256(selected)}


def _parent_verification() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    calls = {
        "claim_matrix": lambda: claim_matrix_gate.verify_report(_load(CLAIM_MATRIX)),
        "v4_final": lambda: v4_gate.verify_report(_load(V4)),
        "v5_final": lambda: v5_gate.validate_report(V5),
        "phase6c_integrity": lambda: phase6c_gate.verify_report(report_path=PHASE6C),
        "board_blocker": lambda: board_gate.verify_report(_load(BOARD_BLOCKER)),
    }
    for key, call in calls.items():
        try:
            call()
            checks[key] = True
        except Exception:
            checks[key] = False
    return checks


def _placements(*allowed: str) -> dict[str, bool]:
    return {key: key in allowed for key in PLACEMENTS}


def _claim(
    claim_id: str,
    source_group: str,
    source_state: str,
    publication_state: str,
    polarity: str,
    wording: str,
    boundary_zh: str,
    grade: str,
    layers: Sequence[str],
    required_layers: Sequence[str],
    placements: Sequence[str],
    reports: Sequence[str],
    raw_data: Sequence[str],
    code: Sequence[str],
    selectors: Sequence[str],
    display_target: str,
    current_result: Any,
    upgrade_or_revocation: str,
    forbidden: Sequence[str],
    lane_id: str | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "source_group": source_group,
        "source_state": source_state,
        "publication_state": publication_state,
        "assertion_polarity": polarity,
        "safe_wording_en": wording,
        "boundary_zh": boundary_zh,
        "evidence_grade": grade,
        "evidence_layers": {"current": list(layers), "required_for_upgrade": list(required_layers)},
        "placements": _placements(*placements),
        "evidence": {
            "reports": list(reports), "raw_data": list(raw_data), "code": list(code),
            "selectors": list(selectors),
        },
        "display_target": display_target,
        "current_result": current_result,
        "upgrade_or_revocation_condition": upgrade_or_revocation,
        "forbidden_wording": list(forbidden),
        "lane_id": lane_id,
    }


V4_POLICY = {
    "CONTRACT_SYSTEM_INTEGRATION": ("ALLOWED_RESTRICTED", "POSITIVE", "PROJECT_NATIVE_PREBOARD", ["PROJECT_NATIVE_SIMULATION", "FIXED_POINT_INTEGER_REFERENCE", "CXXRTL_PREBOARD"], ["BOARD_MEASURED"], ["title", "abstract", "results", "conclusion"], "Fig. 1; Table 1", "仅限预板系统集成与确定性执行合同，不代表新 V5 性能或真板结果。"),
    "SMOOTH_LOCKED_EWMA_ADVANTAGE": ("ALLOWED_RESTRICTED", "POSITIVE", "PROJECT_NATIVE_MATCHED", ["PROJECT_NATIVE_SIMULATION"], [], ["abstract", "results", "conclusion"], "Fig. 3a; Table S2", "只允许报告冻结 smooth matrix 的 aggregate paired 结果；Holm 确认仅 periodic drift。"),
    "STATIC_GKP_SUPERIORITY": ("MANDATORY_NEGATIVE", "NEGATIVE", "NEGATIVE", ["PROJECT_NATIVE_SIMULATION"], [], ["results", "conclusion"], "Fig. 3a; Table S2", "必须保留 static joint MAP 更优的反例，不得隐藏或改写为总体优势。"),
    "STATIC_K4_HARD_ACTION_EQUIVALENCE": ("RESULTS_ONLY", "POSITIVE", "PROJECT_NATIVE_PREBOARD", ["PROJECT_NATIVE_SIMULATION", "FIXED_POINT_INTEGER_REFERENCE"], [], ["results", "supplement"], "Table S3", "仅对冻结 covariance/prior 与完整 10-bit domain 的 hard action 等价成立。"),
    "TAIL_SAFETY_AND_IMPROVEMENT": ("ALLOWED_RESTRICTED", "POSITIVE", "PROJECT_NATIVE_MATCHED", ["PROJECT_NATIVE_SIMULATION"], [], ["results", "conclusion"], "Fig. 3b; Table S4", "结论是预注册 margin 下的 safety/non-inferiority，不是广义 tail-LER 改善。"),
    "GENERAL_DRIFT_EXTERNAL_COMPARISON": ("RESULTS_ONLY", "POSITIVE", "PROJECT_NATIVE_MATCHED", ["PROJECT_NATIVE_SIMULATION"], [], ["results", "supplement"], "Table S5", "BOCD 的 paired LER 结果必须与其 worst-update 预算失败同时报告。"),
    "PUVIANI_NMF_SURPASS": ("PROHIBITED_POSITIVE", "NEGATIVE", "BLOCKED", [], ["OFFICIAL_CODE_REPRODUCTION"], ["conclusion", "supplement", "related_work"], "Table S6 (blocked)", "GQF 源码/协议未闭合，不得声称超过 NMF 寿命。"),
    "FPGA_DETERMINISTIC_ARCHITECTURE": ("ALLOWED_RESTRICTED", "POSITIVE", "PROJECT_NATIVE_PREBOARD", ["FIXED_POINT_INTEGER_REFERENCE", "CXXRTL_PREBOARD", "POST_ROUTE_ESTIMATE"], ["BOARD_MEASURED"], ["abstract", "results", "conclusion"], "Fig. 4a; Table 2", "六周期、II=1、Fmax/资源均为预板或 post-route estimate。"),
    "BOARD_MEASURED_CORRECTNESS_LATENCY": ("BLOCKED", "NEGATIVE", "BLOCKED", [], ["BOARD_MEASURED"], ["conclusion", "supplement"], "Table 2 measured rows (null)", "42 个 measured 字段保持 null；不得用 P&R 数值填充。"),
    "FPGA_SPEED_ADVANTAGE": ("PROHIBITED_POSITIVE", "NEGATIVE", "BLOCKED", [], ["BOARD_MEASURED"], ["conclusion", "related_work"], "Table 2 comparison row (not ranked)", "缺少同任务真板 comparator，不能声称 faster/fastest。"),
    "CNN_AND_HMM_ROLE": ("META_BOUNDARY", "META", "META", ["PROJECT_NATIVE_SIMULATION"], [], ["results", "conclusion", "supplement"], "Fig. 1; ablation Table S7", "legacy CNN residual 仅为消融，HMM 仅为软件慢回路。"),
}


def _v4_claims(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in report["claims"]:
        claim_id = row["claim_id"]
        state, polarity, grade, layers, required, placements, display, boundary = V4_POLICY[claim_id]
        result.append(_claim(
            claim_id, "V4_FINAL_GATE", row["final_state"], state, polarity,
            row["allowed_wording"], boundary, grade, layers, required, placements,
            ["v4_report"], ["v4_source"], ["v4_code"],
            [f"claims.{claim_id}", *[f"upstream:{item['task_id']}:{selector}" for item in (row["evidence"] if isinstance(row["evidence"], list) else [row["evidence"]]) for selector in item["selectors"]]],
            display, row["current_result"], row["remaining_gate"], row["forbidden_wording"],
        ))
    return result


V5_WORDING = {
    "V5-ALG-LER-10PCT": "The preregistered V5 claim of at least 10% LER reduction was revoked at the causal-headroom gate.",
    "V5-TAIL-CALIBRATION-TELEGRAPH": "No V5 calibration/telegraph tail experiment was run after the preregistered early stop.",
    "V5-POSTERIOR-MIXTURE-ACTION": "The proposed V5 posterior-mixture action claim was revoked because incremental action-space headroom was below its entry gate.",
    "V5-UNTOUCHED-FORMAL": "No untouched V5 formal split was accessed or produced.",
    "V5-QUANTIZED-RETENTION": "No V5 quantized-retention result exists because no candidate entered implementation.",
    "V5-LONG-CXXRTL": "The V4 long CXXRTL replay does not qualify a nonexistent V5 design.",
    "V5-FORMAL-ATOMIC-SAFETY": "No V5 formal atomic-safety result exists.",
    "V5-MULTISEED-PR": "No V5 multi-seed place-and-route result exists.",
    "V5-MEASURED-HARDWARE": "V5 board measurement remains blocked and all measured fields remain null.",
    "PHASE6C-READONLY-AUX": "Phase 6C is a read-only secondary evidence layer and cannot rescue or upgrade Phase 6B.",
}

PHASE6C_EVIDENCE = {
    "P6C-SINGLE-CI-CPD-EQUIVALENCE": {
        "reports": ["phase6c_report", "single_cpd_report"], "raw_data": ["phase6c_source", "single_cpd_source"],
        "code": ["phase6c_code", "single_cpd_code"], "upstream_selectors": ["proof_contract", "production_domain", "boundary_audit"],
    },
    "P6C-SURFACE-NOH-CI-ML": {
        "reports": ["phase6c_report", "surface_cnot_report"], "raw_data": ["phase6c_source", "surface_cnot_source"],
        "code": ["phase6c_code", "surface_cnot_code"], "upstream_selectors": ["model_contract", "points", "multiplicity", "boundary_audit"],
    },
    "P6C-STRUCTURED-OFFICIAL-CPD": {
        "reports": ["phase6c_report", "official_cpd_report"], "raw_data": ["phase6c_source", "official_cpd_source"],
        "code": ["phase6c_code", "official_cpd_code"], "upstream_selectors": ["official_import", "independent_experiment", "evidence_boundary"],
    },
    "P6C-MULTIMODE-POSTERIOR-WEIGHTED": {
        "reports": ["phase6c_report", "multimode_report"], "raw_data": ["phase6c_source", "multimode_source"],
        "code": ["phase6c_code", "multimode_code"], "upstream_selectors": ["method_contract", "summaries", "comparisons", "formal_counts"],
    },
    "P6C-LEARNED-ELIGIBILITY": {
        "reports": ["phase6c_report", "learned_report"], "raw_data": ["phase6c_source", "learned_source"],
        "code": ["phase6c_code", "learned_code"], "upstream_selectors": ["required_signature", "candidates", "eligibility_summary"],
    },
    "P6C-NMF-GQF": {
        "reports": ["phase6c_report", "gqf_exact_report", "gqf_gate_report"],
        "raw_data": ["phase6c_source", "gqf_exact_source", "gqf_gate_source"],
        "code": ["phase6c_code", "gqf_exact_code", "gqf_gate_code"],
        "upstream_selectors": ["exact_qualification", "exact_reproduction_status", "prerequisite_ledger", "matched_comparison_metrics"],
    },
    "P6C-AQEC-WALLCLOCK": {
        "reports": ["phase6c_report", "aqec_report"], "raw_data": ["phase6c_source", "aqec_source"],
        "code": ["phase6c_code", "aqec_code"], "upstream_selectors": ["cells", "project_result", "official_protocol_reproduction"],
    },
    "P6C-FPGA-NORMALIZATION": {
        "reports": ["phase6c_report", "preboard_report", "external_fpga_report"],
        "raw_data": ["phase6c_source", "preboard_source", "external_fpga_source"],
        "code": ["phase6c_code", "preboard_code", "external_fpga_code"],
        "upstream_selectors": ["hardware_profiles", "board_measurement_state", "external_rows", "comparison_eligibility", "claim_boundary"],
    },
}


def _v5_claims(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in report["claim_registry"]:
        claim_id = row["claim_id"]
        meta = claim_id == "PHASE6C-READONLY-AUX"
        state = "META_BOUNDARY" if meta else ("BLOCKED" if row["state"] == "BLOCKED" else "PROHIBITED_POSITIVE")
        placements = ["related_work", "supplement"] if meta else ["conclusion", "supplement"]
        result.append(_claim(
            claim_id, "V5_EARLY_STOP", row["state"], state, "META" if meta else "NEGATIVE",
            V5_WORDING[claim_id], "该项服从 T6.10.1 早停，不得用 V4 或 Phase 6C 结果替代。",
            "META" if meta else "BLOCKED", [], ["BOARD_MEASURED"] if claim_id == "V5-MEASURED-HARDWARE" else [], placements,
            ["v5_report"], ["v5_source"], ["v5_code"], [f"claim_registry.{claim_id}"],
            "Boundary Table S1", {"reason": row["reason"]}, "Only a new preregistered route may reopen this claim.",
            ["V5 outperforms", "V5 is state-of-the-art", "V5 hardware result"],
        ))
    return result


def _cell_values(report: Mapping[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    wanted = set(ids)
    return {
        row["cell_id"]: {"metric_id": row["metric_id"], "value": row["value"], "value_state": row["value_state"], "evidence_grade": row["evidence_grade"]}
        for row in report["cells"] if row["cell_id"] in wanted
    }


def _phase6c_claims(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("P6C-SINGLE-CI-CPD-EQUIVALENCE", "single_mode_decoder", "SUPPORTED_NARROW", "RESULTS_ONLY", "POSITIVE", "PROJECT_NATIVE_MATCHED", ["PROJECT_NATIVE_SIMULATION"],
         "For a single-mode square lattice with Euclidean distance, closest-point decoding reduces exactly to closest-integer binning; MAP boundaries remain distinct.",
         "CI=Euclidean CPD 只在冻结的 single-mode square-lattice 条件成立。", ["single_standard_binning_pl"], "Table S8", ["results", "supplement"]),
        ("P6C-SURFACE-NOH-CI-ML", "surface_gkp_gate_outer_code", "PROJECT_NATIVE_MATCHED", "RESULTS_ONLY", "POSITIVE", "PROJECT_NATIVE_MATCHED", ["PROJECT_NATIVE_SIMULATION"],
         "A project-native matched replay of the Noh CNOT model yields lower failure probabilities for ML than CI at 9, 12 and 13 dB.",
         "这是 outer-code/gate lane 的 project-native matched replay，不是本项目 single-mode 主系统排名。", ["surface_ci_9p0db", "surface_ml_9p0db", "surface_ci_12p0db", "surface_ml_12p0db", "surface_ci_13p0db", "surface_ml_13p0db"], "Fig. S1", ["results", "supplement"]),
        ("P6C-STRUCTURED-OFFICIAL-CPD", "multimode_structured_lattice_cpd", "OFFICIAL_REPRODUCTION", "RESULTS_ONLY", "POSITIVE", "OFFICIAL_CODE_REPRODUCTION", ["OFFICIAL_CODE_REPRODUCTION"],
         "The pinned official structured-lattice code reproduces small-distance CPD and analog-MWPM crossings near sigma=0.6025 and 0.5996, respectively.",
         "只报告官方代码的小距离 crossing；不得外推到 single-mode 主任务或更大距离阈值。", ["structured_official_cpd_threshold", "structured_official_analog_mwpm_threshold"], "Fig. S2", ["results", "supplement"]),
        ("P6C-MULTIMODE-POSTERIOR-WEIGHTED", "multimode_structured_lattice_cpd", "PROJECT_NATIVE_MATCHED", "RESULTS_ONLY", "POSITIVE", "PROJECT_NATIVE_MATCHED", ["PROJECT_NATIVE_SIMULATION"],
         "On the frozen multimode secondary split, observed-only posterior-predictive weighted CPD lowers aggregate pL relative to both static Euclidean and weighted-static CPD.",
         "正结果仅属于独立 multimode secondary split，不进入 V5 >=10% 分母或 Route-A 主排名。", ["multimode_static_euclidean_pl", "multimode_weighted_static_pl", "multimode_observed_only_posterior_predictive_weighted_pl"], "Fig. S3", ["results", "supplement"]),
        ("P6C-LEARNED-ELIGIBILITY", "controller_rl_nmf", "ZERO_ELIGIBLE", "MANDATORY_NEGATIVE", "NEGATIVE", "NEGATIVE", ["PROJECT_NATIVE_SIMULATION"],
         "No learned checkpoint satisfies the frozen same-task schema, observation, action and budget eligibility contract.",
         "learned comparator 数量为零，不能用近似网络或旧 checkpoint 填补。", ["controller_learned_same_task"], "Table S9", ["results", "conclusion", "supplement"]),
        ("P6C-NMF-GQF", "controller_rl_nmf", "BLOCKED", "BLOCKED", "NEGATIVE", "BLOCKED", [],
         "Official GQF/NMF reproduction and a same-task lifetime comparison remain blocked; no NMF-surpass conclusion is available.",
         "源码/协议缺口与跨 simulator lifetime 不可比性必须保留。", ["controller_gqf_exact", "controller_gqf_route_a_rank"], "Table S9 (blocked)", ["conclusion", "supplement", "related_work"]),
        ("P6C-AQEC-WALLCLOCK", "aqec_wallclock", "PROJECT_NATIVE_ONLY", "RESULTS_ONLY", "POSITIVE", "PROJECT_NATIVE_MATCHED", ["PROJECT_NATIVE_SIMULATION"],
         "The common-wall-clock AQEC replay is project-native and reports active-versus-autonomous outcomes, while official-protocol reproduction remains blocked.",
         "不能把 project-native 700-us wall-clock replay 写成官方 AQEC 物理复现或零延迟控制。", ["aqec_cutoff12_high_measurement_feedback", "aqec_cutoff12_high_autonomous", "aqec_official_protocol_blocked"], "Fig. S4", ["results", "supplement"]),
        ("P6C-FPGA-NORMALIZATION", "fpga_implementation", "DESCRIPTIVE_ONLY", "RELATED_WORK_ONLY", "META", "LITERATURE_ONLY", ["LITERATURE_ONLY", "POST_ROUTE_ESTIMATE"],
         "The FPGA atlas reports literature values and project post-route estimates descriptively; it contains zero same-task external comparator rows and supports no speed ranking.",
         "18 个外部实现只作规范化 related-work atlas；null 不填补、跨任务不排名。", ["fpga_project_latency_ns", "fpga_project_initiation_interval_ns", "fpga_external_same_task", "fpga_board_latency_null"], "Table S10", ["related_work", "supplement"]),
    ]
    claims = []
    for claim_id, lane, source_state, pub_state, polarity, grade, layers, wording, boundary, cell_ids, display, placements in specs:
        required = ["BOARD_MEASURED"] if claim_id == "P6C-FPGA-NORMALIZATION" else []
        evidence = PHASE6C_EVIDENCE[claim_id]
        upstream_results = {
            artifact_id: {
                "task_id": _load(ARTIFACT_PATHS[artifact_id]).get("task_id"),
                "verdict": _load(ARTIFACT_PATHS[artifact_id]).get("verdict", _load(ARTIFACT_PATHS[artifact_id]).get("status")),
            }
            for artifact_id in evidence["reports"] if artifact_id != "phase6c_report"
        }
        claims.append(_claim(
            claim_id, "PHASE6C_SECONDARY", source_state, pub_state, polarity, wording, boundary,
            grade, layers, required, placements, evidence["reports"], evidence["raw_data"], evidence["code"],
            [*[f"cells.{cell_id}" for cell_id in cell_ids], *[f"upstream.{selector}" for selector in evidence["upstream_selectors"]]],
            display, {"atlas_cells": _cell_values(report, cell_ids), "upstream_reports": upstream_results},
            "A new preregistered same-task lane is required for any broader comparison.",
            ["global winner", "overall SOTA", "faster than all FPGA decoders", "rescues V5"], lane_id=lane,
        ))
    return claims


def _write_source_data(claims: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    fields = [
        "claim_id", "source_group", "source_state", "publication_state", "assertion_polarity",
        "lane_id", "evidence_grade", "current_layers_json", "required_layers_json",
        "placements_json", "report_artifacts_json", "raw_artifacts_json", "code_artifacts_json",
        "artifact_hashes_json", "selectors_json", "display_target", "safe_wording_en", "boundary_zh",
        "current_result_json", "upgrade_or_revocation_condition", "forbidden_wording_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in claims:
            evidence_ids = row["evidence"]["reports"] + row["evidence"]["raw_data"] + row["evidence"]["code"]
            writer.writerow({
                "claim_id": row["claim_id"], "source_group": row["source_group"], "source_state": row["source_state"],
                "publication_state": row["publication_state"], "assertion_polarity": row["assertion_polarity"],
                "lane_id": row["lane_id"] or "", "evidence_grade": row["evidence_grade"],
                "current_layers_json": json.dumps(row["evidence_layers"]["current"], ensure_ascii=False, separators=(",", ":")),
                "required_layers_json": json.dumps(row["evidence_layers"]["required_for_upgrade"], ensure_ascii=False, separators=(",", ":")),
                "placements_json": json.dumps(row["placements"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "report_artifacts_json": json.dumps(row["evidence"]["reports"], separators=(",", ":")),
                "raw_artifacts_json": json.dumps(row["evidence"]["raw_data"], separators=(",", ":")),
                "code_artifacts_json": json.dumps(row["evidence"]["code"], separators=(",", ":")),
                "artifact_hashes_json": json.dumps({key: artifacts[key]["sha256"] for key in evidence_ids}, sort_keys=True, separators=(",", ":")),
                "selectors_json": json.dumps(row["evidence"]["selectors"], ensure_ascii=False, separators=(",", ":")),
                "display_target": row["display_target"], "safe_wording_en": row["safe_wording_en"],
                "boundary_zh": row["boundary_zh"],
                "current_result_json": json.dumps(row["current_result"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "upgrade_or_revocation_condition": row["upgrade_or_revocation_condition"],
                "forbidden_wording_json": json.dumps(row["forbidden_wording"], ensure_ascii=False, separators=(",", ":")),
            })
    temporary.replace(path)


def _render_markdown(claims: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# T7.1.1 claim–evidence–boundary matrix", "",
        "> 论文主论点：本项目建立并验证了一个确定性、fail-closed 的 GKP 双回路预板执行合同，并获得若干严格限定的分任务仿真结果；不声称 V5 LER、NMF 寿命、真实 break-even 或跨 FPGA SOTA。", "",
        "## 使用规则", "",
        "- `ALLOWED_RESTRICTED` 只能逐字保留限定词；`RESULTS_ONLY` 不得提升到标题；`MANDATORY_NEGATIVE/BLOCKED/PROHIBITED_POSITIVE` 必须作为负结果或限制保留。",
        "- `PROJECT_NATIVE_MATCHED`、`OFFICIAL_CODE_REPRODUCTION` 与 `LITERATURE_ONLY` 不得合并为同一排名。",
        "- `POST_ROUTE_ESTIMATE` 不得写成 `BOARD_MEASURED`；T6.9.2 的 42 个 measured 字段继续为 null。", "",
        "## 原子矩阵", "",
        "| Claim | 状态 | 可出现位置 | 证据等级 / 层 | 安全措辞与边界 | 图表 |", "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in claims:
        places = ", ".join(key for key, value in row["placements"].items() if value) or "none"
        layers = ", ".join(row["evidence_layers"]["current"]) or "none"
        wording = row["safe_wording_en"].replace("|", "\\|")
        boundary = row["boundary_zh"].replace("|", "\\|")
        lines.append(f"| `{row['claim_id']}` | {row['publication_state']} | {places} | {row['evidence_grade']} / {layers} | {wording}<br>{boundary} | {row['display_target']} |")
    lines += ["", "## 总结边界", "", "当前可以冻结的是 restricted pre-board contract paper 的 claim contract，而不是完整跨 lane 的高水平性能论文；V5 入口早停和真板阻塞均保持有效。", ""]
    return "\n".join(lines)


def _csv_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["claim_id"] for row in csv.DictReader(handle)]


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    claims = {row["claim_id"]: row for row in report["claims"]}
    artifacts = report["artifact_registry"]
    current_parent = _parent_verification() if check_live_files else report["parent_verification"]
    current_board = _board_binding(BOARD.read_text(encoding="utf-8")) if check_live_files else report["board_status_binding"]
    source_path = ROOT / report["source_data"]["path"]
    markdown_path = ROOT / report["markdown"]["path"]
    v5_source = {row["claim_id"]: row["state"] for row in _load(V5)["claim_registry"]}
    forbidden_positive_phrases = ("state-of-the-art", "outperforms all", "faster than all", "beyond break-even", "surpasses Puviani")
    evidence_ids = {
        key for row in claims.values() for category in ("reports", "raw_data", "code") for key in row["evidence"][category]
    }
    positive = [row for row in claims.values() if row["assertion_polarity"] == "POSITIVE" and row["publication_state"] in {"ALLOWED_RESTRICTED", "RESULTS_ONLY"}]
    blocked = [row for row in claims.values() if row["publication_state"] in {"MANDATORY_NEGATIVE", "PROHIBITED_POSITIVE", "BLOCKED"}]
    measured = _load(BOARD_BLOCKER)["measured_results"]
    return {
        "G01_all_parent_verifiers_pass_live": all(current_parent.values()) and report["parent_verification"] == current_parent,
        "G02_all_artifact_bindings_are_live": set(artifacts) == set(ARTIFACT_PATHS) and evidence_ids <= set(artifacts) and all(len(row["sha256"]) == 64 for row in artifacts.values()) and (not check_live_files or all(_live(row) for row in artifacts.values())),
        "G03_all_v4_claims_are_accounted_once": {key for key, row in claims.items() if row["source_group"] == "V4_FINAL_GATE"} == V4_IDS,
        "G04_all_v5_claims_are_accounted_once": {key for key, row in claims.items() if row["source_group"] == "V5_EARLY_STOP"} == V5_IDS,
        "G05_phase6c_summaries_cover_all_six_lanes": {key for key, row in claims.items() if row["source_group"] == "PHASE6C_SECONDARY"} == PHASE6C_IDS and {row["lane_id"] for row in claims.values() if row["source_group"] == "PHASE6C_SECONDARY"} == LANES,
        "G06_publication_and_placement_schema_is_closed": all(row["publication_state"] in PUBLICATION_STATES and set(row["placements"]) == set(PLACEMENTS) and row["assertion_polarity"] in {"POSITIVE", "NEGATIVE", "META"} for row in claims.values()),
        "G07_revoked_blocked_and_negative_claims_are_not_positive_headline_claims": all(row["assertion_polarity"] != "POSITIVE" and not row["placements"]["title"] and not row["placements"]["abstract"] for row in blocked),
        "G08_every_claim_has_wording_boundary_display_and_evidence_selectors": all(row["safe_wording_en"] and row["boundary_zh"] and row["display_target"] and row["upgrade_or_revocation_condition"] and row["forbidden_wording"] and row["evidence"]["selectors"] for row in claims.values()),
        "G09_every_positive_claim_maps_report_raw_code_and_valid_layers": all(row["evidence"]["reports"] and row["evidence"]["raw_data"] and row["evidence"]["code"] and set(row["evidence_layers"]["current"]) <= EVIDENCE_LAYERS and set(row["evidence_layers"]["required_for_upgrade"]) <= EVIDENCE_LAYERS for row in positive),
        "G10_evidence_grades_and_phase6c_provenance_remain_separate": all(row["evidence_grade"] in EVIDENCE_GRADES for row in claims.values()) and report["provenance_policy"] == {"literature_vs_reproduction_vs_project_native": "SEPARATE", "cross_lane_ranking": "PROHIBITED", "missing_value_imputation": "PROHIBITED"},
        "G11_board_measurement_is_blocked_and_all_fields_are_null": len(measured) == 42 and all(value is None for value in measured.values()) and claims["BOARD_MEASURED_CORRECTNESS_LATENCY"]["publication_state"] == "BLOCKED" and claims["V5-MEASURED-HARDWARE"]["publication_state"] == "BLOCKED" and "BOARD_MEASURED" not in claims["FPGA_DETERMINISTIC_ARCHITECTURE"]["evidence_layers"]["current"],
        "G12_v5_states_match_early_stop_registry_without_resurrection": all(claims[key]["source_state"] == state and claims[key]["assertion_polarity"] != "POSITIVE" for key, state in v5_source.items()),
        "G13_t6_14_3_remains_dropped_and_has_no_result_artifact": report["dropped_task_audit"] == {"task_id": "T6.14.3", "status": "Dropped", "result_artifacts": []},
        "G14_no_global_ranking_or_unqualified_superiority_wording": report["manuscript_decision"]["full_cross_lane_high_level_paper"] == "NO_GO" and report["manuscript_decision"]["allowed_manuscript"] == "RESTRICTED_PREBOARD_CONTRACT_PAPER" and not any(phrase.lower() in row["safe_wording_en"].lower() for row in positive for phrase in forbidden_positive_phrases),
        "G15_terminology_ledger_is_exact_and_required_terms_are_used": report["terminology_ledger"] == TERMINOLOGY_LEDGER and all(term in report["paper_argument"] for term in ("fail-closed", "pre-board", "V5 LER", "NMF lifetime", "FPGA SOTA")),
        "G16_source_csv_is_lossless_hash_bound_one_row_per_claim": report["source_data"]["rows"] == len(claims) == 29 and source_path.is_file() and len(_csv_ids(source_path)) == len(claims) and set(_csv_ids(source_path)) == set(claims) and (not check_live_files or _live(report["source_data"])),
        "G17_human_matrix_is_live_and_contains_every_claim": markdown_path.is_file() and all(f"`{key}`" in markdown_path.read_text(encoding="utf-8") for key in claims) and (not check_live_files or _live(report["markdown"])),
        "G18_board_status_binding_is_semantic_and_live": report["board_status_binding"] == current_board and current_board["statuses"] == {"T6.14.3": "Dropped", "T6.15.5": "Done", "T6.19.3": "Done", "T7.1.1": "ACTIVE_OR_DONE"},
        "G19_one_substantive_mutation_per_gate_fails_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 19 and len(report["semantic_mutation_audit"]["cases"]) == 19,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def mutate_claim(value: dict[str, Any], claim_id: str) -> dict[str, Any]:
        return next(row for row in value["claims"] if row["claim_id"] == claim_id)

    def attempt(name: str, target: str, change: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 19, "detected": 19, "cases": []}
        change(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[target]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": target, "rejected": rejected})

    attempt("parent_failure", "G01_all_parent_verifiers_pass_live", lambda x: x["parent_verification"].update(v5_final=False))
    attempt("artifact_hash_forge", "G02_all_artifact_bindings_are_live", lambda x: x["artifact_registry"]["v4_report"].update(sha256="0"))
    attempt("drop_v4_claim", "G03_all_v4_claims_are_accounted_once", lambda x: x["claims"].remove(mutate_claim(x, "CONTRACT_SYSTEM_INTEGRATION")))
    attempt("drop_v5_claim", "G04_all_v5_claims_are_accounted_once", lambda x: x["claims"].remove(mutate_claim(x, "V5-ALG-LER-10PCT")))
    attempt("mislabel_secondary_lane", "G05_phase6c_summaries_cover_all_six_lanes", lambda x: mutate_claim(x, "P6C-AQEC-WALLCLOCK").update(lane_id="single_mode_decoder"))
    attempt("unknown_publication_state", "G06_publication_and_placement_schema_is_closed", lambda x: mutate_claim(x, "CONTRACT_SYSTEM_INTEGRATION").update(publication_state="SOTA"))
    attempt("promote_blocked_to_abstract", "G07_revoked_blocked_and_negative_claims_are_not_positive_headline_claims", lambda x: mutate_claim(x, "V5-ALG-LER-10PCT")["placements"].update(abstract=True))
    attempt("remove_boundary", "G08_every_claim_has_wording_boundary_display_and_evidence_selectors", lambda x: mutate_claim(x, "SMOOTH_LOCKED_EWMA_ADVANTAGE").update(boundary_zh=""))
    attempt("remove_positive_raw", "G09_every_positive_claim_maps_report_raw_code_and_valid_layers", lambda x: mutate_claim(x, "CONTRACT_SYSTEM_INTEGRATION")["evidence"].update(raw_data=[]))
    attempt("merge_provenance", "G10_evidence_grades_and_phase6c_provenance_remain_separate", lambda x: x["provenance_policy"].update(literature_vs_reproduction_vs_project_native="MERGED"))
    attempt("promote_estimate_to_measured", "G11_board_measurement_is_blocked_and_all_fields_are_null", lambda x: mutate_claim(x, "FPGA_DETERMINISTIC_ARCHITECTURE")["evidence_layers"]["current"].append("BOARD_MEASURED"))
    attempt("resurrect_v5", "G12_v5_states_match_early_stop_registry_without_resurrection", lambda x: mutate_claim(x, "V5-ALG-LER-10PCT").update(assertion_polarity="POSITIVE"))
    attempt("invent_t6_14_3_result", "G13_t6_14_3_remains_dropped_and_has_no_result_artifact", lambda x: x["dropped_task_audit"].update(result_artifacts=["fake.json"]))
    attempt("promote_full_paper", "G14_no_global_ranking_or_unqualified_superiority_wording", lambda x: x["manuscript_decision"].update(full_cross_lane_high_level_paper="GO"))
    attempt("rename_canonical_term", "G15_terminology_ledger_is_exact_and_required_terms_are_used", lambda x: x["terminology_ledger"].update(primary_system="CNN-centric system"))
    attempt("forge_csv_rows", "G16_source_csv_is_lossless_hash_bound_one_row_per_claim", lambda x: x["source_data"].update(rows=28))
    attempt("disconnect_markdown", "G17_human_matrix_is_live_and_contains_every_claim", lambda x: x["markdown"].update(path="docs/nonexistent_claim_matrix.md"))
    attempt("change_dropped_status", "G18_board_status_binding_is_semantic_and_live", lambda x: x["board_status_binding"]["statuses"].update({"T6.14.3": "Done"}))
    attempt("forge_mutation_count", "G19_one_substantive_mutation_per_gate_fails_closed", lambda x: x.update(semantic_mutation_audit={"count": 19, "detected": 18, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report(source_data: Path = DEFAULT_SOURCE_DATA, markdown: Path = DEFAULT_MARKDOWN) -> dict[str, Any]:
    v4 = _load(V4)
    v5 = _load(V5)
    phase6c = _load(PHASE6C)
    artifacts = {key: _binding(path) for key, path in ARTIFACT_PATHS.items()}
    claims = _v4_claims(v4) + _v5_claims(v5) + _phase6c_claims(phase6c)
    _write_source_data(claims, artifacts, source_data)
    _atomic_text(_render_markdown(claims), markdown)
    task_statuses = _task_statuses(BOARD.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_argument": "The Route-A contract system establishes a deterministic, fail-closed pre-board GKP dual-loop execution contract with restricted task-specific simulation evidence; it does not establish V5 LER superiority, NMF lifetime superiority, board-measured break-even, or cross-FPGA SOTA.",
        "terminology_ledger": TERMINOLOGY_LEDGER,
        "evidence_layer_ontology": {
            "LITERATURE_ONLY": "reported external value without local reproduction",
            "OFFICIAL_CODE_REPRODUCTION": "locally rerun pinned official implementation",
            "PROJECT_NATIVE_SIMULATION": "project-native matched simulation",
            "FIXED_POINT_INTEGER_REFERENCE": "bit-accurate integer software reference",
            "CXXRTL_PREBOARD": "cycle-accurate pre-board RTL simulation",
            "POST_ROUTE_ESTIMATE": "multi-seed synthesis/place-and-route estimate",
            "BOARD_MEASURED": "same-bitstream physical-board measurement",
        },
        "provenance_policy": {"literature_vs_reproduction_vs_project_native": "SEPARATE", "cross_lane_ranking": "PROHIBITED", "missing_value_imputation": "PROHIBITED"},
        "artifact_registry": artifacts,
        "parent_verification": _parent_verification(),
        "board_status_binding": _board_binding(BOARD.read_text(encoding="utf-8")),
        "dropped_task_audit": {
            "task_id": "T6.14.3", "status": task_statuses["T6.14.3"],
            "result_artifacts": sorted(_relative(path) for pattern in ("t6_14_3*.json", "t6_14_3*.csv") for path in (ROOT / "docs").glob(pattern)),
        },
        "claims": claims,
        "claim_coverage": {"v4": sorted(V4_IDS), "v5": sorted(V5_IDS), "phase6c": sorted(PHASE6C_IDS), "total": len(claims)},
        "manuscript_decision": {
            "full_cross_lane_high_level_paper": "NO_GO",
            "allowed_manuscript": "RESTRICTED_PREBOARD_CONTRACT_PAPER",
            "abstract_claim_ids": sorted(row["claim_id"] for row in claims if row["placements"]["abstract"]),
            "conclusion_claim_ids": sorted(row["claim_id"] for row in claims if row["placements"]["conclusion"]),
            "mandatory_negative_claim_ids": sorted(row["claim_id"] for row in claims if row["publication_state"] in {"MANDATORY_NEGATIVE", "PROHIBITED_POSITIVE", "BLOCKED"}),
        },
        "source_data": {**_binding(source_data), "rows": len(claims)},
        "markdown": _binding(markdown),
    }
    report["semantic_mutation_audit"] = {"count": 19, "detected": 19, "cases": []}
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": [key for key, value in report["gates"].items() if not value]}
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_CLAIM_BOUNDARY_INTEGRITY"
    report["analysis_sha256"] = _canonical_sha256({key: report[key] for key in ("paper_argument", "terminology_ledger", "evidence_layer_ontology", "provenance_policy", "artifact_registry", "parent_verification", "board_status_binding", "dropped_task_audit", "claims", "claim_coverage", "manuscript_decision", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")})
    return report


def verify_report(report: Mapping[str, Any] | None = None, path: Path = DEFAULT_REPORT) -> dict[str, bool]:
    value = dict(report) if report is not None else _load(path)
    gates = evaluate_gates(value)
    expected_hash = _canonical_sha256({key: value[key] for key in ("paper_argument", "terminology_ledger", "evidence_layer_ontology", "provenance_policy", "artifact_registry", "parent_verification", "board_status_binding", "dropped_task_audit", "claims", "claim_coverage", "manuscript_decision", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")})
    checks = {
        "identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION,
        "gates": value.get("gates") == gates and all(gates.values()),
        "verdict": value.get("verdict") == VERDICT,
        "analysis_hash": value.get("analysis_sha256") == expected_hash,
    }
    if not all(checks.values()):
        raise ValueError(f"T7.1.1 verification failed: {[key for key, passed in checks.items() if not passed]}")
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
    print(json.dumps({"output": _relative(args.report), "claims": len(report["claims"]), "gates": report["gate_summary"], "verdict": report["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
