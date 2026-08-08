"""Freeze and validate the T6.5.1 Route-A claim/role/lane contract."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs/t6_5_1_route_a_claim_contract.json"
DEFAULT_CSV = ROOT / "docs/t6_5_1_route_a_claim_contract_source_data.csv"
VERDICT = "PASS_ROUTE_A_CLAIM_ROLE_AND_LANE_CONTRACT_FROZEN"

SOURCE_PATHS = (
    "docs/claim_ladder.json",
    "docs/decoder_controller_terminology.json",
    "docs/t5_1_4_algorithm_branch_verdict.json",
    "docs/t4_4_5_teacher_student_branch_freeze.json",
    "docs/t5_5_2_target_device_synthesis.json",
    "docs/t6_2_2_long_rtl_qualification.json",
    "docs/experiment_plan.md",
)

SUPPORTED = {"SUPPORTED_CONTRACT", "SUPPORTED_BOUNDED", "SUPPORTED_EXTENSION_ONLY"}
FUTURE_OR_LIMITED = {"CONDITIONAL_FUTURE", "PROHIBITED_NOW", "ABLATION_ONLY"}
CLAIM_STATUSES = SUPPORTED | FUTURE_OR_LIMITED
EVIDENCE_LEVELS = (
    "governance_contract",
    "software_simulation",
    "bit_exact_python",
    "rtl_cxxrtl",
    "synthesis_estimate",
    "place_route_estimate",
    "board_measured",
    "physical_device_measured",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _role(
    role_id: str,
    canonical_term: str,
    responsibility: str,
    output_authority: str,
    *,
    role_type: str = "required_component",
    replaceable: bool = False,
    deployability: str = "deployable",
) -> dict[str, Any]:
    return {
        "role_id": role_id,
        "canonical_term": canonical_term,
        "responsibility": responsibility,
        "output_authority": output_authority,
        "role_type": role_type,
        "replaceable": replaceable,
        "deployability": deployability,
    }


def build_contract() -> dict[str, Any]:
    roles = [
        _role(
            "ROLE-SYSTEM",
            "regime-aware safe adaptive dual-loop",
            "paper-level system that composes performance, safety, atomic update and deterministic execution contracts",
            "system orchestration only; it does not own a new decoder score",
            role_type="primary_system",
            deployability="target_deployable_after_contract_and_promotion",
        ),
        _role(
            "ROLE-MAP",
            "static/adaptive joint MAP decoder",
            "produce logical-coset LLR/decision and own decoder-lane LER",
            "logical decision and MAP-LUT candidate image",
        ),
        _role(
            "ROLE-REGIME",
            "causal regime posterior estimator",
            "infer normal/smooth/shift/burst/leakage posterior from observed history",
            "update permission/freeze/trusted-bank proposal; never the logical action",
        ),
        _role(
            "ROLE-EVENT",
            "event and leakage/reset FSM",
            "convert accepted observations and safety state into hold/recovery/reset modes",
            "event mode, reset request and hysteresis state",
        ),
        _role(
            "ROLE-FALLBACK",
            "conservative fallback and health monitor",
            "fail closed on uncertainty, integrity, freshness or deadline faults",
            "fallback/rollback/reason/health; never a better-than-oracle decision",
        ),
        _role(
            "ROLE-BANK",
            "versioned trusted A/B MAP bank",
            "stage, validate, atomically commit and roll back last-known-good parameter images",
            "bank/version/commit acknowledgement",
        ),
        _role(
            "ROLE-RTL",
            "FPGA fast-path executor",
            "execute the latched fixed-point MAP/event/action contract with deterministic core timing",
            "registered action/state words and core-cycle evidence, not host estimation",
        ),
        _role(
            "ROLE-CNN",
            "legacy CNN residual module",
            "optional learned slow-loop proposal under matched budget",
            "bounded proposal only; never primary system or unqualified LER owner",
            role_type="replaceable_learning_extension",
            replaceable=True,
            deployability="candidate_after_matched_gate",
        ),
        _role(
            "ROLE-TEACHER",
            "Feedback-GRAPE/NMF teacher",
            "offline high-capacity controller target inside the controller lane",
            "training target/control residual; not a decoder oracle",
            role_type="replaceable_learning_extension",
            replaceable=True,
            deployability="offline_privileged_training_only",
        ),
        _role(
            "ROLE-STUDENT",
            "distilled low-dimensional student",
            "optional bounded controller recurrence distilled from a teacher",
            "controller extension action subject to safety wrapper",
            role_type="replaceable_learning_extension",
            replaceable=True,
            deployability="candidate_after_retention_and_safety_gate",
        ),
        _role(
            "ROLE-ORACLE",
            "hidden-state decoder/model oracle",
            "provide an isolated nondeployable upper reference",
            "upper-bound decision/gap only; excluded from deployable aggregates",
            role_type="privileged_reference",
            replaceable=False,
            deployability="nondeployable_hidden_truth",
        ),
    ]

    lanes = [
        {
            "lane_id": "LANE-DECODER",
            "canonical_name": "same-trace GKP decoder lane",
            "data_domain": "protocol-aligned syndrome stream",
            "privilege": "observed_only_deployable_plus_isolated_hidden_oracle",
            "allowed_metrics": [
                "per_round_pL", "per_round_pX", "per_round_pY", "per_round_pZ",
                "average_LER", "p95_LER", "worst_window_LER",
                "static_to_oracle_gap_closure", "adaptation_lag",
                "false_update", "unnecessary_fallback", "avoided_errors",
                "induced_errors", "decoder_MAC", "decoder_memory", "update_rate",
            ],
            "forbidden_comparisons": [
                "GQF logical lifetime minus syndrome LER",
                "privileged oracle inside deployable aggregate",
                "raw LER copied from a different squeezing/noise/round definition",
            ],
        },
        {
            "lane_id": "LANE-GQF",
            "canonical_name": "official GQF controller lane",
            "data_domain": "fixed-commit official Matteo-Puviani/GQF environment",
            "privilege": "same_simulator_same_training_selection_and_seed_budget",
            "allowed_metrics": [
                "GQF_TX", "GQF_TY", "GQF_TZ", "GQF_Tch", "GQF_Favg",
                "GQF_lifetime_gain_retention", "controller_params", "controller_MAC",
                "controller_memory", "unsafe_control_action", "controller_fallback",
            ],
            "forbidden_comparisons": [
                "project syndrome LER relabelled as GQF lifetime",
                "project T2.3.7 directional ranking relabelled as official reproduction",
                "surpass NMF before paper-exact reproduction and paired lifetime gate",
            ],
        },
        {
            "lane_id": "LANE-HARDWARE",
            "canonical_name": "task-normalized FPGA QEC hardware lane",
            "data_domain": "matched code/problem/precision/latency-boundary hardware task",
            "privilege": "evidence_level_explicit",
            "allowed_metrics": [
                "core_latency_cycles", "core_latency_time", "source_to_action_latency",
                "closed_loop_latency", "initiation_interval", "deadline_miss",
                "Fmax", "LUT", "FF", "BRAM", "DSP", "estimated_power", "measured_power",
            ],
            "forbidden_comparisons": [
                "surface-code raw decoder latency as direct single-mode GKP ranking",
                "P&R estimate relabelled as measured board latency",
                "core-only latency compared with source-to-action latency",
            ],
        },
    ]

    def claim(
        claim_id: str,
        lane_id: str,
        status: str,
        wording_en: str,
        wording_zh: str,
        metrics: list[str],
        data_domain: str,
        privilege: str,
        required_evidence: str,
        current_evidence: list[str],
        activation_gate: list[str],
        revocation_conditions: list[str],
        forbidden_wording: list[str],
    ) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "lane_id": lane_id,
            "support_status": status,
            "canonical_wording_en": wording_en,
            "canonical_wording_zh": wording_zh,
            "metrics": metrics,
            "data_domain": data_domain,
            "privilege": privilege,
            "required_evidence_level": required_evidence,
            "current_evidence": current_evidence,
            "activation_gate": activation_gate,
            "revocation_conditions": revocation_conditions,
            "forbidden_wording": forbidden_wording,
        }

    claims = [
        claim(
            "CLAIM-ARCH-01", "LANE-GOVERNANCE", "SUPPORTED_CONTRACT",
            "We define a contract-centric, regime-aware safe adaptive dual-loop in which MAP decoding owns logical-error performance, regime/event/fallback logic owns tail safety, and the FPGA fast path owns deterministic execution.",
            "本项目已冻结 contract-centric、regime-aware 安全自适应双回路：MAP 负责逻辑错误性能，regime/event/fallback 负责 tail 安全，FPGA fast path 负责确定性执行。",
            [], "cross-lane role and evidence contract", "governance_only",
            "governance_contract",
            ["docs/experiment_plan.md"],
            ["T6.5.1 contract validator passes"],
            ["a component is allowed to claim another component's metric", "lane nonmixing is removed"],
            ["one model solves decoding, safety and hardware latency", "global leaderboard"],
        ),
        claim(
            "CLAIM-RTL-QUAL-01", "LANE-HARDWARE", "SUPPORTED_BOUNDED",
            "The production fast-path core is cycle-exact against an independent integer reference over one million board-independent CXXRTL qualification cycles.",
            "production fast-path core 已在板卡无关 CXXRTL 中以独立整数 reference 完成百万周期全字段 cycle-exact 资格验证。",
            ["core_latency_cycles", "initiation_interval"], "production core qualification trace",
            "observed_integer_words_no_physical_transport", "rtl_cxxrtl",
            ["docs/t6_2_2_long_rtl_qualification.json"],
            ["T6.2.2 all gates pass"],
            ["source/trace hash changes without rerun", "any bit mismatch or undefined action appears"],
            ["board validated", "transport qualified", "measured FPGA latency"],
        ),
        claim(
            "CLAIM-FPGA-EST-01", "LANE-HARDWARE", "SUPPORTED_BOUNDED",
            "The current fixed-point fast path has a target-device-specific three-seed place-and-route timing and resource estimate; it is not a board measurement.",
            "当前定点 fast path 具有目标器件、三 seed 的 P&R 时序/资源估计，但不是板测。",
            ["Fmax", "LUT", "FF", "BRAM", "DSP", "core_latency_cycles"],
            "Tang Nano 20K target-device estimate", "tool_report_only", "place_route_estimate",
            ["docs/t5_5_2_target_device_synthesis.json"],
            ["source/constraint hashes and all three P&R seeds pass"],
            ["tool report or constraint hash changes", "estimate is presented without target/clock qualifier"],
            ["measured power", "board timing", "vendor signoff"],
        ),
        claim(
            "CLAIM-STATIC-01", "LANE-DECODER", "CONDITIONAL_FUTURE",
            "[Conditional; not currently supported] Under a shared protocol-aligned trace and compute contract, Route-A reduces aggregate and tail logical error relative to the strongest deployable static GKP decoder.",
            "仅当同 trace/预算证据通过时，Route-A 才可声称相对最强可部署 static GKP decoder 改善 aggregate/tail LER。",
            ["average_LER", "p95_LER", "worst_window_LER", "static_to_oracle_gap_closure"],
            "unified syndrome benchmark", "observed_only_deployable", "software_simulation",
            [], ["T6.7.1 and T6.7.2 pass", "T6.8.1 same-trace paired CI supports the claim"],
            ["aggregate paired 95% LCB is not positive", "any preregistered tail gate fails"],
            ["universal GKP superiority", "surface-code threshold", "state-of-the-art GKP decoder"],
        ),
        claim(
            "CLAIM-DRIFT-01", "LANE-DECODER", "CONDITIONAL_FUTURE",
            "[Conditional; not currently supported] Under matched history, update cadence and compute budget, Route-A improves drift adaptation without violating preregistered abrupt/OOD safety margins.",
            "仅在 matched history/cadence/预算与 OOD 安全门通过后，才声称 Route-A 具有一般漂移适应优势。",
            ["average_LER", "worst_window_LER", "adaptation_lag", "false_update", "unnecessary_fallback"],
            "held-out smooth and abrupt/OOD syndrome scenarios", "observed_only_deployable", "software_simulation",
            [], ["T6.7.1, T6.7.2 and T6.7.4 pass", "T6.8.2 includes at least one reproducible external method"],
            ["external implementation is unavailable", "catastrophic-degradation or nominal non-inferiority gate fails"],
            ["adaptive decoding is universally superior", "all noise regimes"],
        ),
        claim(
            "CLAIM-TAIL-01", "LANE-DECODER", "CONDITIONAL_FUTURE",
            "[Conditional; not currently supported] Regime-triggered freeze, trusted-bank switching and fail-closed fallback reduce harmful tail events without unacceptable nominal fallback cost.",
            "仅当 abrupt/OOD 与 nominal non-inferiority 门同时通过时，才声称 regime freeze/trusted-bank/fallback 改善 tail 安全。",
            ["p95_LER", "worst_window_LER", "avoided_errors", "induced_errors", "false_update", "unnecessary_fallback"],
            "unified syndrome benchmark", "observed_only_deployable", "software_simulation",
            [], ["T6.7.2 independent formal evaluation passes", "T6.7.4 promotion gate passes"],
            ["calibration-shift counterexample persists", "nominal fallback cost exceeds frozen margin"],
            ["guarantees safety", "eliminates catastrophic errors"],
        ),
        claim(
            "CLAIM-CNN-01", "LANE-DECODER", "ABLATION_ONLY",
            "The legacy CNN residual is a replaceable matched-budget ablation and is not required for the Route-A system claim.",
            "legacy CNN residual 仅是 matched-budget 可替换消融，不是 Route-A 成立前提。",
            ["average_LER", "p95_LER", "worst_window_LER", "decoder_MAC", "decoder_memory"],
            "unified syndrome benchmark", "observed_only_candidate", "software_simulation",
            ["docs/t5_1_4_algorithm_branch_verdict.json"],
            ["may be promoted only by its own T6.6/T6.7 matched gate"],
            ["matched comparison fails", "checkpoint is absent or cost is unmatched"],
            ["CNN is generally optimal", "CNN drives the primary contribution", "learned decoder is universally superior"],
        ),
        claim(
            "CLAIM-STUDENT-01", "LANE-GOVERNANCE", "SUPPORTED_EXTENSION_ONLY",
            "A low-dimensional student retains a preregistered fraction of teacher gain in the existing bounded project controller simulator, but this is neither official GQF reproduction nor decoder evidence.",
            "低维 student 在既有受限 controller simulator 中具有预注册 teacher gain-retention 证据，但不等于官方 GQF 复现，也不是 decoder 证据。",
            [],
            "project bounded finite-cutoff controller simulator", "offline_teacher_and_candidate_student", "software_simulation",
            ["docs/t4_4_5_teacher_student_branch_freeze.json"],
            ["existing T4.4.5 bounded branch remains current"],
            ["official GQF is implied", "long-horizon/OOD retention is implied", "controller result is used as decoder evidence"],
            ["official GQF reproduction", "surpasses Puviani NMF", "RNN decoder"],
        ),
        claim(
            "CLAIM-GQF-01", "LANE-GQF", "PROHIBITED_NOW",
            "[Not currently claimable] In the same official GQF environment and budget, the Route-A extension improves logical-channel lifetime over Puviani NMF.",
            "[当前禁止] 在同一官方 GQF 环境与预算下超过 Puviani NMF 的 logical-channel lifetime。",
            ["GQF_TX", "GQF_TY", "GQF_TZ", "GQF_Tch", "GQF_Favg"],
            "official fixed-commit GQF simulator", "same_budget_official_comparison", "software_simulation",
            [], ["T6.8.3 intake passes", "T6.8.4 paper-exact reproduction passes", "T6.8.5 paired lifetime 95% LCB > 0"],
            ["absolute values or ordering miss reproduction tolerance", "paired lifetime LCB is not positive"],
            ["currently surpasses NMF", "Puviani lifetime record"],
        ),
        claim(
            "CLAIM-HW-SPEED-01", "LANE-HARDWARE", "PROHIBITED_NOW",
            "[Not currently claimable] The Route-A FPGA decoder is faster than existing FPGA QEC decoders on a matched task boundary.",
            "[当前禁止] Route-A FPGA decoder 在同任务边界下比现有 FPGA QEC decoder 更快。",
            ["source_to_action_latency", "closed_loop_latency", "initiation_interval", "deadline_miss"],
            "task-normalized real-board comparison", "board_measured_same_boundary", "board_measured",
            [], ["T6.8.6 comparable subset is non-null", "T6.9.2 same-bitstream board measurement passes"],
            ["only core/P&R estimate is available", "code/problem/precision/latency boundary differs", "any board deadline miss occurs"],
            ["fastest FPGA decoder", "measured speed advantage", "real-time board result"],
        ),
        claim(
            "CLAIM-BREAK-EVEN-01", "LANE-GOVERNANCE", "PROHIBITED_NOW",
            "[Not currently claimable] The system exceeds physical break-even or a measured physical-memory lifetime record.",
            "[当前禁止] 系统超过真实物理 break-even 或物理存储寿命记录。",
            [], "physical oscillator experiment", "physical_device_measurement", "physical_device_measured",
            [], ["protocol-aligned physical-device experiment with registered passive reference and lifetime uncertainty"],
            ["evidence is simulation, CXXRTL, P&R or board-only digital control"],
            ["real break-even", "exceeds Sivak 2023", "physical lifetime record"],
        ),
    ]

    terminology = [
        {"canonical": "regime-aware safe adaptive dual-loop", "variants": ["Route A", "safe adaptive dual-loop"], "rule": "paper-level primary system"},
        {"canonical": "static/adaptive joint MAP decoder", "variants": ["MAP", "adaptive MAP"], "rule": "owns decoder LER"},
        {"canonical": "causal regime posterior estimator", "variants": ["HMM", "regime detector"], "rule": "owns posterior, not logical action"},
        {"canonical": "FPGA fast-path executor", "variants": ["fast path", "RTL core"], "rule": "owns registered execution timing, not host estimation"},
        {"canonical": "hidden-state decoder/model oracle", "variants": ["oracle MAP", "model oracle"], "rule": "always qualified and nondeployable"},
        {"canonical": "Feedback-GRAPE/NMF teacher", "variants": ["teacher", "NMF teacher"], "rule": "controller training target, never RNN decoder/oracle"},
    ]
    return {
        "schema_version": "t6.5.1-route-a-claim-contract-v1",
        "task_id": "T6.5.1",
        "one_sentence_argument": (
            "Under one deployable execution contract, MAP owns average logical-error performance; "
            "regime/event/fallback logic owns tail safety; RTL owns deterministic execution; and "
            "learning modules remain replaceable extensions whose claims activate only through their own gates."
        ),
        "primary_system_role_id": "ROLE-SYSTEM",
        "roles": roles,
        "terminology_ledger": terminology,
        "comparison_lanes": lanes,
        "cross_lane_rule": {
            "global_leaderboard_allowed": False,
            "global_score_allowed": False,
            "raw_cross_simulator_lifetime_subtraction_allowed": False,
            "raw_cross_code_latency_ranking_allowed": False,
            "oracle_in_deployable_aggregate_allowed": False,
        },
        "claims": claims,
        "required_future_tasks": [
            "T6.5.2", "T6.5.3", "T6.6.1", "T6.6.2", "T6.6.3",
            "T6.7.1", "T6.7.2", "T6.7.3", "T6.7.4",
            "T6.8.1", "T6.8.2", "T6.8.3", "T6.8.4", "T6.8.5", "T6.8.6", "T6.8.7",
            "T6.9.1", "T6.9.2", "T6.9.3",
        ],
    }


def load_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    bindings = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        bindings.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size})
        if path.suffix == ".json":
            evidence[relative] = json.loads(path.read_text(encoding="utf-8"))
    evidence["source_bindings"] = bindings
    return evidence


def validate_contract(contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, bool]:
    roles = list(contract["roles"])
    role_by_id = {role["role_id"]: role for role in roles}
    lanes = list(contract["comparison_lanes"])
    lane_by_id = {lane["lane_id"]: lane for lane in lanes}
    claims = list(contract["claims"])
    t514 = evidence["docs/t5_1_4_algorithm_branch_verdict.json"]
    t445 = evidence["docs/t4_4_5_teacher_student_branch_freeze.json"]
    t552 = evidence["docs/t5_5_2_target_device_synthesis.json"]
    t622 = evidence["docs/t6_2_2_long_rtl_qualification.json"]
    forbidden_canonical = (
        "surface-code threshold", "global leaderboard", "global sota",
        "cnn is universally", "fastest fpga decoder",
    )
    evaluation_metric_sets = [set(lane["allowed_metrics"]) for lane in lanes]
    supported_claims = [claim for claim in claims if claim["support_status"] in SUPPORTED]
    current_paths_ok = all(
        all((ROOT / path).is_file() for path in claim["current_evidence"])
        for claim in supported_claims
    )
    claim_fields = {
        "claim_id", "lane_id", "support_status", "canonical_wording_en",
        "canonical_wording_zh", "metrics", "data_domain", "privilege",
        "required_evidence_level", "current_evidence", "activation_gate",
        "revocation_conditions", "forbidden_wording",
    }
    lanes_plus_governance = set(lane_by_id) | {"LANE-GOVERNANCE"}
    claim_metrics_valid = all(
        not claim["metrics"]
        if claim["lane_id"] == "LANE-GOVERNANCE"
        else set(claim["metrics"]).issubset(set(lane_by_id[claim["lane_id"]]["allowed_metrics"]))
        for claim in claims
    )
    gqf = next(claim for claim in claims if claim["claim_id"] == "CLAIM-GQF-01")
    hw_speed = next(claim for claim in claims if claim["claim_id"] == "CLAIM-HW-SPEED-01")
    cnn = next(claim for claim in claims if claim["claim_id"] == "CLAIM-CNN-01")
    break_even = next(claim for claim in claims if claim["claim_id"] == "CLAIM-BREAK-EVEN-01")
    return {
        "source_artifacts_exist_and_are_hash_bound": len(evidence["source_bindings"]) == len(SOURCE_PATHS) and all(len(row["sha256"]) == 64 for row in evidence["source_bindings"]),
        "exactly_one_primary_system": sum(role["role_type"] == "primary_system" for role in roles) == 1 and contract["primary_system_role_id"] == "ROLE-SYSTEM",
        "canonical_roles_complete_and_unique": len(roles) == 11 and len(role_by_id) == len(roles) and all(role.get("responsibility") and role.get("output_authority") for role in roles),
        "learning_modules_are_replaceable_extensions": all(role["replaceable"] and role["role_type"] == "replaceable_learning_extension" for role in roles if role["role_id"] in ("ROLE-CNN", "ROLE-TEACHER", "ROLE-STUDENT")),
        "oracle_is_isolated_and_nondeployable": role_by_id["ROLE-ORACLE"]["role_type"] == "privileged_reference" and role_by_id["ROLE-ORACLE"]["deployability"] == "nondeployable_hidden_truth",
        "map_regime_event_fallback_rtl_authorities_do_not_overlap": (
            "logical decision" in role_by_id["ROLE-MAP"]["output_authority"]
            and "never the logical action" in role_by_id["ROLE-REGIME"]["output_authority"]
            and "event mode" in role_by_id["ROLE-EVENT"]["output_authority"]
            and "reason" in role_by_id["ROLE-FALLBACK"]["output_authority"]
            and "registered action" in role_by_id["ROLE-RTL"]["output_authority"]
        ),
        "exactly_three_evaluation_lanes": len(lanes) == 3 and set(lane_by_id) == {"LANE-DECODER", "LANE-GQF", "LANE-HARDWARE"},
        "lane_metric_namespaces_are_disjoint": all(evaluation_metric_sets[i].isdisjoint(evaluation_metric_sets[j]) for i in range(3) for j in range(i + 1, 3)),
        "global_ranking_and_oracle_mixing_are_forbidden": all(value is False for value in contract["cross_lane_rule"].values()),
        "claims_are_schema_complete_and_unique": len(claims) == 11 and len({claim["claim_id"] for claim in claims}) == len(claims) and all(claim_fields.issubset(claim) for claim in claims),
        "claim_lanes_metrics_status_and_evidence_levels_are_valid": all(claim["lane_id"] in lanes_plus_governance and claim["support_status"] in CLAIM_STATUSES and claim["required_evidence_level"] in EVIDENCE_LEVELS for claim in claims) and claim_metrics_valid,
        "every_claim_has_activation_revocation_and_forbidden_wording": all(claim["activation_gate"] and claim["revocation_conditions"] and claim["forbidden_wording"] for claim in claims),
        "supported_current_claims_have_existing_evidence": current_paths_ok,
        "t5_negative_cnn_branch_is_preserved": t514["fallback_contract"]["branch_id"] == "event_aware_adaptive_map_fpga_codesign" and t514["fallback_contract"]["cnn_or_learned_performance_claim_retained"] is False and cnn["support_status"] == "ABLATION_ONLY",
        "teacher_student_is_bounded_extension_not_decoder_or_official_gqf": (
            t445["active_branch"]["branch_id"] == "qualified_student_retention"
            and next(claim for claim in claims if claim["claim_id"] == "CLAIM-STUDENT-01")["support_status"] == "SUPPORTED_EXTENSION_ONLY"
            and next(claim for claim in claims if claim["claim_id"] == "CLAIM-STUDENT-01")["lane_id"] == "LANE-GOVERNANCE"
        ),
        "rtl_qualification_is_supported_but_board_claim_is_closed": all(t622["gates"].values()) and t622["verdict"].startswith("PASS_BOARD_INDEPENDENT") and t552["evidence_boundary"]["board_measured"] is False and t552["evidence_boundary"]["transport_implemented"] is False,
        "gqf_surpass_hardware_speed_and_break_even_are_prohibited_now": gqf["support_status"] == hw_speed["support_status"] == break_even["support_status"] == "PROHIBITED_NOW",
        "canonical_wording_lint_rejects_unbounded_terms": not any(token in claim["canonical_wording_en"].lower() for claim in claims for token in forbidden_canonical),
        "future_task_chain_is_complete": set(contract["required_future_tasks"]) == {f"T6.{major}.{minor}" for major, max_minor in ((5, 3), (6, 3), (7, 4), (8, 7), (9, 3)) for minor in range(1, max_minor + 1)} - {"T6.5.1"},
    }


def semantic_mutation_audit(contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    mutations: list[dict[str, Any]] = []

    def attempt(name: str, mutator: Any) -> None:
        candidate = copy.deepcopy(contract)
        mutator(candidate)
        mutations.append({"mutation": name, "rejected": not all(validate_contract(candidate, evidence).values())})

    def role(candidate: dict[str, Any], role_id: str) -> dict[str, Any]:
        return next(row for row in candidate["roles"] if row["role_id"] == role_id)

    def claim(candidate: dict[str, Any], claim_id: str) -> dict[str, Any]:
        return next(row for row in candidate["claims"] if row["claim_id"] == claim_id)

    attempt("make_cnn_primary", lambda value: role(value, "ROLE-CNN").update(role_type="primary_system", replaceable=False))
    attempt("make_oracle_deployable", lambda value: role(value, "ROLE-ORACLE").update(deployability="deployable"))
    attempt("let_regime_emit_logical_action", lambda value: role(value, "ROLE-REGIME").update(output_authority="logical decision"))
    attempt("allow_global_leaderboard", lambda value: value["cross_lane_rule"].update(global_leaderboard_allowed=True))
    attempt("mix_gqf_metric_into_decoder", lambda value: next(row for row in value["comparison_lanes"] if row["lane_id"] == "LANE-DECODER")["allowed_metrics"].append("GQF_Tch"))
    attempt("promote_gqf_without_reproduction", lambda value: claim(value, "CLAIM-GQF-01").update(support_status="SUPPORTED_BOUNDED"))
    attempt("promote_hardware_speed_without_board", lambda value: claim(value, "CLAIM-HW-SPEED-01").update(support_status="SUPPORTED_BOUNDED"))
    attempt("erase_claim_revocation", lambda value: claim(value, "CLAIM-TAIL-01").update(revocation_conditions=[]))
    attempt("insert_surface_threshold_wording", lambda value: claim(value, "CLAIM-STATIC-01").update(canonical_wording_en="We establish a surface-code threshold."))
    attempt("remove_future_gate_task", lambda value: value["required_future_tasks"].remove("T6.9.2"))
    return {"mutations": mutations, "count": len(mutations), "detected": sum(row["rejected"] for row in mutations)}


def write_source_data(path: Path, claims: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "claim_id", "lane_id", "support_status", "required_evidence_level", "data_domain",
        "privilege", "metrics", "canonical_wording_en", "canonical_wording_zh",
        "current_evidence", "activation_gate", "revocation_conditions", "forbidden_wording",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in claims:
            writer.writerow({
                key: " | ".join(map(str, row[key])) if isinstance(row[key], list) else row[key]
                for key in fields
            })


def run_contract(
    *,
    artifact_path: Path = DEFAULT_JSON,
    source_data_path: Path = DEFAULT_CSV,
) -> dict[str, Any]:
    contract = build_contract()
    evidence = load_evidence()
    gates = validate_contract(contract, evidence)
    audit = semantic_mutation_audit(contract, evidence)
    gates["semantic_mutations_rejected"] = audit["count"] == audit["detected"] == 10
    report = {
        **contract,
        "source_bindings": evidence["source_bindings"],
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "failed": [key for key, value in gates.items() if not value],
        },
        "semantic_mutation_audit": audit,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "verdict": VERDICT if all(gates.values()) else "FAIL_ROUTE_A_CLAIM_CONTRACT",
        "implementation_sha256": _sha256(Path(__file__)),
        "source_data": {"path": _relative(source_data_path), "row_count": len(contract["claims"])},
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    write_source_data(source_data_path, contract["claims"])
    report["source_data"]["sha256"] = _sha256(source_data_path)
    artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    report = run_contract(artifact_path=args.artifact, source_data_path=args.source_data)
    print(json.dumps({"verdict": report["verdict"], "gate_summary": report["gate_summary"], "claims": len(report["claims"]), "roles": len(report["roles"]), "lanes": len(report["comparison_lanes"]), "artifact": _relative(args.artifact)}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
