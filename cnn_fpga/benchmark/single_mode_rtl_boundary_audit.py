"""T6.25.1 live-source and task-boundary audit for the single-mode RTL lane."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase6d/t6_25_1_single_mode_rtl_boundary.json"
REPORT = ROOT / "docs/t6_25_1_single_mode_rtl_boundary_audit.json"
SOURCE_DATA = ROOT / "docs/t6_25_1_single_mode_rtl_boundary_audit_source_data.csv"
MARKDOWN = ROOT / "docs/single_mode_rtl_boundary_audit.md"
RUNNER = Path(__file__).resolve()

EXPECTED_PARENT_VERDICTS = {
    "T6.2.1": "PASS_PRODUCTION_RTL_SHELL_READY_FOR_T6_2_2_LONG_TRACE",
    "T6.2.2": "PASS_BOARD_INDEPENDENT_LONG_RTL_QUALIFICATION_READY_FOR_ROUTE_A",
    "T6.7.3": "PASS_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION",
    "T6.9.1": "PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED",
    "T6.19.1": "PASS_STATIC_MAP_LUT_PREBOARD_PROFILE_OTHERS_NA_HOST_STAGES_SEPARATE",
    "T6.20.2": "PASS_DUAL_LANE_CONTRACT_FROZEN",
}


class IntegrityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing binding: {path}")
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return (
        path.is_file()
        and _sha256(path) == binding["sha256"]
        and path.stat().st_size == int(binding["bytes"])
    )


def _source_hashes_live(source_hashes: Mapping[str, str]) -> tuple[bool, list[str]]:
    stale = [rel for rel, digest in source_hashes.items() if not (ROOT / rel).is_file() or _sha256(ROOT / rel) != digest]
    return not stale, stale


def _parent_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id, rel in config["parent_reports"].items():
        path = ROOT / rel
        payload = _load(path)
        row: dict[str, Any] = {
            "artifact": _binding(path),
            "verdict": payload.get("verdict"),
            "expected_verdict": EXPECTED_PARENT_VERDICTS[task_id],
            "verdict_matches": payload.get("verdict") == EXPECTED_PARENT_VERDICTS[task_id],
        }
        gates = payload.get("gates", {})
        if isinstance(gates, dict):
            row["reported_gates_pass"] = bool(gates) and all(bool(value) for value in gates.values())
        else:
            row["reported_gates_pass"] = False
        if task_id == "T6.7.3":
            live, stale = _source_hashes_live(payload["source_hashes"])
            row.update({"direct_source_bindings": len(payload["source_hashes"]), "direct_sources_live": live, "stale_sources": stale})
            anchor = payload["t6_2_2_anchor"]
            row["parent_anchor_live"] = _sha256(ROOT / anchor["path"]) == anchor["sha256"]
        elif task_id == "T6.9.1":
            bindings = payload["source_bindings"]
            stale = [entry["path"] for entry in bindings if not _binding_live(entry)]
            row.update({"direct_source_bindings": len(bindings), "direct_sources_live": not stale, "stale_sources": stale})
        elif task_id == "T6.19.1":
            bindings = list(payload["bindings"].values())
            stale = [entry["path"] for entry in bindings if not _binding_live(entry)]
            row.update({"direct_source_bindings": len(bindings), "direct_sources_live": not stale, "stale_sources": stale})
        elif task_id == "T6.2.2":
            trace = ROOT / payload["trace"]["path"]
            row.update({
                "direct_source_bindings": 0,
                "direct_sources_live": None,
                "trace_live": trace.is_file() and _sha256(trace) == payload["trace"]["sha256"],
                "source_binding_gap": "legacy report binds generated CXXRTL model/trace but not the input source files",
            })
        elif task_id == "T6.2.1":
            row.update({
                "direct_source_bindings": 0,
                "direct_sources_live": None,
                "source_binding_gap": "legacy report records source paths and generated model hash but not input source hashes",
            })
        else:
            bindings = list(payload.get("artifact_registry", {}).values())
            stale = [entry["path"] for entry in bindings if not _binding_live(entry)]
            row.update({"direct_source_bindings": len(bindings), "direct_sources_live": not stale, "stale_sources": stale})
        result[task_id] = row
    return result


def _module_audit(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = set(config["required_converged_capabilities"])
    rows: list[dict[str, Any]] = []
    for item in config["current_tops"]:
        path = ROOT / item["path"]
        source = path.read_text(encoding="utf-8")
        declaration_count = len(re.findall(rf"\bmodule\s+{re.escape(item['module'])}\b", source))
        child_hits = {
            child: len(
                re.findall(
                    rf"\b{re.escape(child)}\b\s*(?:#\s*\([\s\S]*?\)\s*)?[A-Za-z_][A-Za-z0-9_]*\s*\(",
                    source,
                )
            )
            for child in item["children"]
        }
        capabilities = set(item["capabilities"])
        rows.append({
            **item,
            "source": _binding(path),
            "module_declaration_count": declaration_count,
            "child_instantiation_hits": child_hits,
            "children_present": all(count >= 1 for count in child_hits.values()),
            "missing_converged_capabilities": sorted(required - capabilities),
            "is_converged_production_top": required <= capabilities,
        })
    return rows


def _token_audit(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rel, tokens in config["required_source_tokens"].items():
        source = (ROOT / rel).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in source]
        rows.append({"path": rel, "required_tokens": tokens, "missing_tokens": missing, "passed": not missing})
    return rows


def _transitive_rtl_scope(module_rows: list[dict[str, Any]]) -> dict[str, Any]:
    module_names = {row["module"] for row in module_rows}
    child_names = {child for row in module_rows for child in row["children"]}
    allowed_external = {"gkp_fast_path_core", "route_a_policy_overlay", "low_dimensional_student_kernel"}
    graph_names = module_names | child_names
    forbidden_pattern = re.compile(r"multimode|coset|matching|mwpm|kmwm|surface_square", re.IGNORECASE)
    hits = sorted(name for name in graph_names if forbidden_pattern.search(name))
    forbidden_source_hits: list[dict[str, str]] = []
    for row in module_rows:
        source = (ROOT / row["path"]).read_text(encoding="utf-8")
        for token in ("multimode", "logical_coset", "logical-coset", "surface_square", "mwpm", "kmwm"):
            if re.search(rf"\b{re.escape(token)}\b", source, flags=re.IGNORECASE):
                forbidden_source_hits.append({"path": row["path"], "token": token})
    return {
        "module_names": sorted(graph_names),
        "all_children_accounted": child_names <= module_names | allowed_external,
        "forbidden_multimode_module_hits": hits,
        "forbidden_multimode_source_hits": forbidden_source_hits,
        "contains_multimode_graph_or_exact_mld": bool(hits or forbidden_source_hits),
    }


def _structural_findings(module_rows: list[dict[str, Any]]) -> dict[str, bool]:
    by_name = {row["module"]: row for row in module_rows}
    production = (ROOT / by_name["gkp_fast_path_production_top"]["path"]).read_text(encoding="utf-8")
    qualification = (ROOT / by_name["gkp_fast_path_qualification_top"]["path"]).read_text(encoding="utf-8")
    integrated = (ROOT / by_name["route_a_integrated_qualification_top"]["path"]).read_text(encoding="utf-8")
    pareto = (ROOT / by_name["route_a_hardware_pareto_synth_top"]["path"]).read_text(encoding="utf-8")
    return {
        "production_instantiates_core": by_name["gkp_fast_path_production_top"]["child_instantiation_hits"]["gkp_fast_path_core"] == 1,
        "production_does_not_instantiate_policy": "route_a_policy_overlay policy" not in production,
        "qualification_instantiates_core_not_production": by_name["gkp_fast_path_qualification_top"]["child_instantiation_hits"]["gkp_fast_path_core"] == 1 and "gkp_fast_path_production_top" not in qualification,
        "integrated_instantiates_core_and_policy_not_production": all(count == 1 for count in by_name["route_a_integrated_qualification_top"]["child_instantiation_hits"].values()) and "gkp_fast_path_production_top" not in integrated,
        "integrated_exposes_raw_cfg_and_trust": all(token in integrated for token in ("input  wire          cfg_we", "input  wire          bank0_trusted", "input  wire          bank1_trusted")),
        "pareto_wraps_integrated_not_production": by_name["route_a_hardware_pareto_synth_top"]["child_instantiation_hits"]["route_a_integrated_qualification_top"] == 1 and "gkp_fast_path_production_top" not in pareto,
    }


def _reuse_decisions(parents: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": "T6.2.1",
            "decision": "REFERENCE_ONLY_NOT_REUSABLE_FOR_CONVERGED_TOP",
            "reason": "production management was exercised for 1,681 cycles without a policy overlay and the legacy report lacks direct source hashes",
            "required_action": "T6.25.2 must prove the converged top",
        },
        {
            "task_id": "T6.2.2",
            "decision": "CORE_LONG_RUN_REGRESSION_ONLY",
            "reason": "the million-cycle wrapper drives raw cfg/trust/commit pins and does not instantiate gkp_fast_path_production_top",
            "required_action": "T6.25.3 must rerun one million or more cycles on the converged top",
        },
        {
            "task_id": "T6.7.3",
            "decision": "POLICY_CORE_LONG_RUN_REGRESSION_ONLY" if parents["T6.7.3"]["direct_sources_live"] else "STALE_RERUN_REQUIRED",
            "reason": "the live policy+core top bypasses production CRC staging and trust ownership",
            "required_action": "retain as regression; do not transfer its PASS to atomic production management",
        },
        {
            "task_id": "T6.9.1",
            "decision": "OLD_HARNESS_PR_REFERENCE_ONLY" if parents["T6.9.1"]["direct_sources_live"] else "STALE_RERUN_REQUIRED",
            "reason": "three-seed P&R is live for route_a_hardware_pareto_synth_top, which wraps the raw-pin qualification top",
            "required_action": "T6.25.4 must rerun three seeds on the converged top",
        },
        {
            "task_id": "T6.19.1",
            "decision": "STATIC_CORE_PROFILE_REFERENCE_ONLY" if parents["T6.19.1"]["direct_sources_live"] else "STALE_RERUN_REQUIRED",
            "reason": "the only eligible hardware row profiles gkp_fast_path_synth_top, not production management plus policy",
            "required_action": "preserve as scoped baseline; never label it the converged Route-A top",
        },
    ]


def _validate_report(report: Mapping[str, Any], *, check_files: bool = True) -> None:
    _require(report["task_id"] == "T6.25.1", "wrong task")
    _require(report["verdict"] == "PASS_BOUNDARY_FROZEN_CONVERGED_PRODUCTION_TOP_REQUIRED", "wrong verdict")
    _require(report["evidence_lane"] == "SINGLE_MODE_DETERMINISTIC_RTL", "wrong lane")
    _require(len(report["module_audit"]) == 5, "module inventory changed")
    _require(all(row["module_declaration_count"] == 1 and row["children_present"] for row in report["module_audit"]), "module graph mismatch")
    required = set(report["required_converged_capabilities"])
    _require(all(row["missing_converged_capabilities"] == sorted(required - set(row["capabilities"])) for row in report["module_audit"]), "capability gap not independently derived")
    _require(all(row["is_converged_production_top"] == (required <= set(row["capabilities"])) for row in report["module_audit"]), "converged flag not independently derived")
    _require(not any(row["is_converged_production_top"] for row in report["module_audit"]), "existing top falsely marked converged")
    _require(report["convergence_gap"]["present"] is True, "convergence gap erased")
    _require(report["convergence_gap"]["next_task"] == "T6.25.2", "gap routed incorrectly")
    _require(report["transitive_rtl_scope"]["contains_multimode_graph_or_exact_mld"] is False, "multimode implementation promoted into RTL")
    _require(report["transitive_rtl_scope"]["forbidden_multimode_module_hits"] == [], "forbidden multimode module hit hidden")
    _require(report["transitive_rtl_scope"]["forbidden_multimode_source_hits"] == [], "forbidden multimode source hit hidden")
    _require(report["transitive_rtl_scope"]["all_children_accounted"] is True, "unknown child module")
    _require(all(row["passed"] for row in report["required_token_audit"]), "source token audit failed")
    _require(all(report["structural_findings"].values()), "production/policy top separation changed")
    _require(all(row["verdict_matches"] and row["reported_gates_pass"] for row in report["parent_evidence"].values()), "parent verdict/gate mismatch")
    _require(report["parent_evidence"]["T6.7.3"]["direct_sources_live"] is True, "T6.7.3 sources stale")
    _require(report["parent_evidence"]["T6.9.1"]["direct_sources_live"] is True, "T6.9.1 sources stale")
    _require(report["parent_evidence"]["T6.19.1"]["direct_sources_live"] is True, "T6.19.1 sources stale")
    _require(report["parent_evidence"]["T6.20.2"]["direct_sources_live"] is True, "T6.20.2 contract sources stale")
    _require(report["parent_evidence"]["T6.2.1"]["direct_source_bindings"] == 0, "legacy T6.2.1 source bindings fabricated")
    _require(report["parent_evidence"]["T6.2.2"]["direct_source_bindings"] == 0, "legacy T6.2.2 source bindings fabricated")
    _require(report["parent_evidence"]["T6.2.2"]["trace_live"] is True, "T6.2.2 raw trace stale")
    _require(len(report["allowed_contract_bridges"]) == 4, "bridge whitelist changed")
    _require(report["claim_boundary"]["board_measurement"] is None, "board measurement fabricated")
    _require(report["claim_boundary"]["fastest_or_speed_advantage"] is False, "fastest claim promoted")
    _require(report["claim_boundary"]["multimode_decoder_deployed_in_rtl"] is False, "multimode deployment fabricated")
    decisions = {row["task_id"]: row["decision"] for row in report["reuse_decisions"]}
    _require(decisions["T6.2.2"] == "CORE_LONG_RUN_REGRESSION_ONLY", "raw-pin long run overpromoted")
    _require(decisions["T6.9.1"] == "OLD_HARNESS_PR_REFERENCE_ONLY", "old P&R overpromoted")
    if check_files:
        _require(all(_binding_live(row) for row in report["bindings"]), "live binding mismatch")


def _semantic_mutations(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    mutations: list[tuple[str, Any]] = []

    def add(name: str, mutator: Any) -> None:
        candidate = copy.deepcopy(report)
        mutator(candidate)
        try:
            _validate_report(candidate, check_files=False)
        except (IntegrityError, KeyError, TypeError, ValueError):
            mutations.append((name, True))
        else:
            mutations.append((name, False))

    add("promote_existing_top", lambda x: x["module_audit"][0].__setitem__("is_converged_production_top", True))
    add("forge_capability_gap", lambda x: x["module_audit"][0].__setitem__("missing_converged_capabilities", []))
    add("erase_gap", lambda x: x["convergence_gap"].__setitem__("present", False))
    add("reroute_gap", lambda x: x["convergence_gap"].__setitem__("next_task", "T6.25.4"))
    add("inject_multimode", lambda x: x["transitive_rtl_scope"].__setitem__("contains_multimode_graph_or_exact_mld", True))
    add("unknown_child", lambda x: x["transitive_rtl_scope"].__setitem__("all_children_accounted", False))
    add("hide_multimode_source", lambda x: x["transitive_rtl_scope"].__setitem__("forbidden_multimode_source_hits", [{"path": "rtl.sv", "token": "multimode"}]))
    add("drop_module", lambda x: x["module_audit"].pop())
    add("duplicate_declaration", lambda x: x["module_audit"][0].__setitem__("module_declaration_count", 2))
    add("remove_child", lambda x: x["module_audit"][1].__setitem__("children_present", False))
    add("token_failure", lambda x: x["required_token_audit"][0].__setitem__("passed", False))
    add("merge_tops_without_rerun", lambda x: x["structural_findings"].__setitem__("integrated_instantiates_core_and_policy_not_production", False))
    add("stale_parent", lambda x: x["parent_evidence"]["T6.7.3"].__setitem__("direct_sources_live", False))
    add("fabricate_legacy_binding", lambda x: x["parent_evidence"]["T6.2.2"].__setitem__("direct_source_bindings", 9))
    add("erase_trace", lambda x: x["parent_evidence"]["T6.2.2"].__setitem__("trace_live", False))
    add("promote_old_long_run", lambda x: next(r for r in x["reuse_decisions"] if r["task_id"] == "T6.2.2").__setitem__("decision", "FULL_REUSE"))
    add("promote_old_pr", lambda x: next(r for r in x["reuse_decisions"] if r["task_id"] == "T6.9.1").__setitem__("decision", "FULL_REUSE"))
    add("fabricate_board", lambda x: x["claim_boundary"].__setitem__("board_measurement", 222.222))
    add("claim_fastest", lambda x: x["claim_boundary"].__setitem__("fastest_or_speed_advantage", True))
    add("claim_multimode_deployed", lambda x: x["claim_boundary"].__setitem__("multimode_decoder_deployed_in_rtl", True))
    return [{"mutation": name, "caught": caught} for name, caught in mutations]


def build_report() -> dict[str, Any]:
    config = _load(CONFIG)
    parents = _parent_evidence(config)
    modules = _module_audit(config)
    tokens = _token_audit(config)
    scope = _transitive_rtl_scope(modules)
    structural_findings = _structural_findings(modules)
    convergence_gap = {
        "present": not any(row["is_converged_production_top"] for row in modules),
        "missing_top": "single top combining production CRC/staging/CAS/drain ownership with Route-A policy/LKG and a target synthesis surface",
        "next_task": "T6.25.2",
        "required_sequence": [
            "construct one converged synthesizable production top",
            "prove atomic/fail-closed properties with reachable covers and mutation kills",
            "rerun >=1e6-cycle independent-golden/CXXRTL on that exact top",
            "rerun three-seed synthesis/P&R on that exact top",
        ],
    }
    report: dict[str, Any] = {
        "task_id": "T6.25.1",
        "schema_version": "t6.25.1-single-mode-rtl-boundary-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_lane": config["evidence_lane"],
        "required_converged_capabilities": config["required_converged_capabilities"],
        "module_audit": modules,
        "required_token_audit": tokens,
        "transitive_rtl_scope": scope,
        "structural_findings": structural_findings,
        "parent_evidence": parents,
        "reuse_decisions": _reuse_decisions(parents),
        "allowed_contract_bridges": config["allowed_contract_bridges"],
        "forbidden_rtl_implementations": config["forbidden_rtl_implementations"],
        "convergence_gap": convergence_gap,
        "claim_boundary": config["claim_boundary"],
        "bindings": [
            _binding(CONFIG),
            _binding(RUNNER),
            *[_binding(ROOT / item["path"]) for item in config["current_tops"]],
            _binding(ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"),
            _binding(ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv"),
            *[_binding(ROOT / rel) for rel in config["parent_reports"].values()],
        ],
        "verdict": "PASS_BOUNDARY_FROZEN_CONVERGED_PRODUCTION_TOP_REQUIRED",
    }
    _validate_report(report)
    report["semantic_mutation_audit"] = _semantic_mutations(report)
    gates = [
        {"gate": "all_parent_verdicts_and_reported_gates_live", "passed": all(row["verdict_matches"] and row["reported_gates_pass"] for row in parents.values())},
        {"gate": "t6_7_3_direct_sources_live", "passed": parents["T6.7.3"]["direct_sources_live"]},
        {"gate": "t6_9_1_direct_sources_live", "passed": parents["T6.9.1"]["direct_sources_live"]},
        {"gate": "t6_19_1_and_t6_20_2_direct_sources_live", "passed": parents["T6.19.1"]["direct_sources_live"] and parents["T6.20.2"]["direct_sources_live"]},
        {"gate": "legacy_unbound_sources_not_fabricated", "passed": parents["T6.2.1"]["direct_source_bindings"] == parents["T6.2.2"]["direct_source_bindings"] == 0},
        {"gate": "t6_2_2_raw_trace_live", "passed": parents["T6.2.2"]["trace_live"]},
        {"gate": "module_graph_and_tokens_match", "passed": all(row["children_present"] and row["module_declaration_count"] == 1 for row in modules) and all(row["passed"] for row in tokens)},
        {"gate": "production_policy_top_separation_detected", "passed": all(structural_findings.values())},
        {"gate": "multimode_implementation_absent", "passed": not scope["contains_multimode_graph_or_exact_mld"]},
        {"gate": "converged_top_gap_detected", "passed": convergence_gap["present"]},
        {"gate": "old_top_evidence_not_promoted", "passed": all("FULL_REUSE" not in row["decision"] for row in report["reuse_decisions"])},
        {"gate": "four_contract_bridges_only", "passed": len(config["allowed_contract_bridges"]) == 4},
        {"gate": "board_and_fastest_claims_closed", "passed": config["claim_boundary"]["board_measurement"] is None and not config["claim_boundary"]["fastest_or_speed_advantage"]},
        {"gate": "all_live_bindings_match", "passed": all(_binding_live(row) for row in report["bindings"])},
        {"gate": "semantic_mutations_caught", "passed": all(row["caught"] for row in report["semantic_mutation_audit"])},
    ]
    report["gates"] = gates
    report["gate_summary"] = {"passed": sum(bool(row["passed"]) for row in gates), "total": len(gates)}
    _require(report["gate_summary"]["passed"] == report["gate_summary"]["total"], "boundary audit gates failed")
    canonical = copy.deepcopy(report)
    canonical.pop("generated_at_utc", None)
    report["analysis_sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return report


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module in report["module_audit"]:
        rows.append({"section": "module", "key": module["module"], "metric": "is_converged", "value": module["is_converged_production_top"], "detail": module["role"]})
        for capability in report["required_converged_capabilities"]:
            rows.append({"section": "capability", "key": module["module"], "metric": capability, "value": capability in module["capabilities"], "detail": module["source"]["path"]})
    for task_id, parent in report["parent_evidence"].items():
        rows.append({"section": "parent", "key": task_id, "metric": "verdict_matches", "value": parent["verdict_matches"], "detail": parent["artifact"]["sha256"]})
        rows.append({"section": "parent", "key": task_id, "metric": "direct_sources_live", "value": parent.get("direct_sources_live"), "detail": parent.get("source_binding_gap", "hash-bound")})
    for decision in report["reuse_decisions"]:
        rows.append({"section": "reuse", "key": decision["task_id"], "metric": "decision", "value": decision["decision"], "detail": decision["reason"]})
    for bridge in report["allowed_contract_bridges"]:
        rows.append({"section": "bridge", "key": bridge["bridge_id"], "metric": "payload", "value": bridge["payload"], "detail": bridge["deployment_implication"]})
    for gate in report["gates"]:
        rows.append({"section": "gate", "key": gate["gate"], "metric": "passed", "value": gate["passed"], "detail": report["verdict"]})
    for mutation in report["semantic_mutation_audit"]:
        rows.append({"section": "mutation", "key": mutation["mutation"], "metric": "caught", "value": mutation["caught"], "detail": "fail-closed"})
    for binding in report["bindings"]:
        rows.append({"section": "binding", "key": binding["path"], "metric": "sha256", "value": binding["sha256"], "detail": binding["bytes"]})
    return rows


def _write_outputs(report: Mapping[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["section", "key", "metric", "value", "detail"])
        writer.writeheader()
        writer.writerows(_source_rows(report))
    modules = "\n".join(
        f"| `{row['module']}` | {row['role']} | {', '.join(row['capabilities'])} | {', '.join(row['missing_converged_capabilities'])} |"
        for row in report["module_audit"]
    )
    reuse = "\n".join(
        f"| {row['task_id']} | `{row['decision']}` | {row['reason']} |" for row in report["reuse_decisions"]
    )
    text = f"""# T6.25.1 single-mode RTL 边界与 live-source 审计

## 结论

**`{report['verdict']}`**，{report['gate_summary']['passed']}/{report['gate_summary']['total']} gates、{len(report['semantic_mutation_audit'])}/{len(report['semantic_mutation_audit'])} semantic mutations 通过。

当前没有一个 top 同时包含 production CRC32/staging/CAS/drain、Route-A policy/LKG 和 target synthesis surface。旧证据本身没有被否定，但只能作为各自旧 top 的 regression/reference；T6.25.2 必须先形成一个 converged production top，后续 property、百万周期 CXXRTL 和三种子 P&R 必须针对同一 top 重新执行。

## 当前 top 能力矩阵

| module | 实际角色 | 已有能力 | 相对 converged top 缺口 |
| --- | --- | --- | --- |
{modules}

关键区别：`gkp_fast_path_production_top` 有完整管理面但没有 policy/LKG；`route_a_integrated_qualification_top` 有 policy/LKG，却直接驱动 core 的 raw `cfg_we` 与 `bank*_trusted`，没有实例化 production management；当前 P&R harness 又包裹后者。因此不能把 T6.2.1、T6.7.3、T6.9.1 的 PASS 横向拼接成同一 actual-top 的 atomic/fail-closed 证明。

## 父证据复用决定

| task | 决定 | 原因 |
| --- | --- | --- |
{reuse}

T6.7.3 的 9 个 direct source hashes、T6.9.1 的 12 个 source bindings、T6.19.1 的 13 个 bindings 当前全部 live；T6.2.1/T6.2.2 的旧报告没有直接绑定输入 source hash，已如实保留这个缺口。T6.2.2 的 1,000,000-cycle raw trace hash仍 live，但它只证明 raw-pin core wrapper。

## 与 multimode 软件 lane 的隔离

RTL transitive inventory 中没有 multimode syndrome graph、logical-coset summation、posterior-predictive integration 或 matching 模块。只允许共享四类 transaction contract：candidate image envelope、atomic active view、single-mode regime command、event/action word。接口名相似不构成 multimode decoder 已部署的证据。

## 下一步强制顺序

1. T6.25.2 构建唯一 converged synthesizable production top，并对该 top 完成 property/cover/mutation；
2. T6.25.3 对完全相同的 top 做每 family 至少 100k、aggregate 至少 1M 的 independent-golden/CXXRTL；
3. T6.25.4 对完全相同的 top 做三种子 synthesis/P&R；
4. 真板 latency/jitter/deadline/power 继续为 null，不能声称 fastest。
"""
    MARKDOWN.write_text(text, encoding="utf-8")


def verify() -> dict[str, Any]:
    stored = _load(REPORT)
    rebuilt = build_report()
    for payload in (stored, rebuilt):
        payload.pop("generated_at_utc", None)
    _require(stored == rebuilt, "stored boundary report differs from live rebuild")
    _require(SOURCE_DATA.is_file() and SOURCE_DATA.stat().st_size > 0, "source data missing")
    _require(MARKDOWN.is_file() and MARKDOWN.stat().st_size > 0, "markdown missing")
    return {"verdict": stored["verdict"], "gates": stored["gate_summary"], "analysis_sha256": stored["analysis_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    report = build_report()
    _write_outputs(report)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "report": _relative(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
