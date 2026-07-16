from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "decoder_controller_terminology.json"
MARKDOWN_PATH = ROOT / "docs" / "decoder_controller_terminology.md"


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _terms(registry: dict) -> dict[str, dict]:
    return {term["term_id"]: term for term in registry["terms"]}


def _bindings(registry: dict) -> dict[str, dict]:
    return {binding["binding_id"]: binding for binding in registry["artifact_bindings"]}


def test_schema_freezes_all_seven_decision_dimensions(registry: dict) -> None:
    assert registry["task_id"] == "T1.4.5"
    assert registry["status"] == "frozen_contract"
    assert registry["decision_dimensions"] == [
        "decision_question",
        "information_set",
        "output_contract",
        "objective",
        "causality",
        "horizon",
        "deployability",
    ]


def test_term_ids_names_and_contract_fields_are_unique_and_complete(registry: dict) -> None:
    terms = registry["terms"]
    assert len(terms) == 11
    assert len({term["term_id"] for term in terms}) == len(terms)
    assert len({term["canonical_name"] for term in terms}) == len(terms)

    required = {
        "term_id",
        "canonical_name",
        "chinese_label",
        "family",
        "definition",
        "decision_question",
        "information_set",
        "output_contract",
        "objective",
        "causality",
        "horizon",
        "deployability",
        "current_status",
        "current_binding_ids",
        "future_tasks",
        "allowed_labels",
        "forbidden_labels",
        "comparison_role",
    }
    for term in terms:
        assert set(term) == required, term["term_id"]
        for field in (
            "canonical_name",
            "chinese_label",
            "family",
            "definition",
            "decision_question",
            "objective",
            "causality",
            "horizon",
            "deployability",
            "current_status",
            "comparison_role",
        ):
            assert term[field]
        assert term["information_set"]
        assert term["output_contract"]
        assert term["allowed_labels"]
        assert term["forbidden_labels"]


def test_markdown_and_json_term_ids_are_exactly_synchronized(registry: dict) -> None:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    markdown_ids = re.findall(r"<!-- term-id: ([A-Z0-9-]+) -->", markdown)
    json_ids = [term["term_id"] for term in registry["terms"]]
    assert len(markdown_ids) == len(set(markdown_ids))
    assert set(markdown_ids) == set(json_ids)
    assert "## 10. 非 demo 审计结论" in markdown


def test_all_artifact_bindings_match_current_source_lines(registry: dict) -> None:
    terms = _terms(registry)
    bindings = _bindings(registry)
    assert len(bindings) == len(registry["artifact_bindings"]) == 20

    referenced = set()
    for term in terms.values():
        for binding_id in term["current_binding_ids"]:
            assert binding_id in bindings, (term["term_id"], binding_id)
            assert bindings[binding_id]["term_id"] == term["term_id"]
            referenced.add(binding_id)
    assert referenced == set(bindings)

    required_fields = {
        "binding_id",
        "term_id",
        "path",
        "line_start",
        "line_end",
        "expected_fragment",
        "implementation_status",
        "notes",
    }
    for binding in bindings.values():
        assert set(binding) == required_fields
        path = ROOT / binding["path"]
        assert path.is_file(), binding["binding_id"]
        lines = path.read_text(encoding="utf-8").splitlines()
        start = binding["line_start"]
        end = binding["line_end"]
        assert 1 <= start <= end <= len(lines)
        anchored = "\n".join(lines[start - 1 : end])
        assert binding["expected_fragment"] in anchored, binding["binding_id"]


def test_three_upper_references_have_distinct_questions_outputs_and_roles(registry: dict) -> None:
    terms = _terms(registry)
    oracle_ids = [
        "TERM-DECODER-ORACLE",
        "TERM-CONTROL-ORACLE",
        "TERM-RECOVERY-BOUND",
    ]
    selected = [terms[term_id] for term_id in oracle_ids]
    assert {term["family"] for term in selected} == {
        "decoder_bound",
        "control_bound",
        "channel_bound",
    }
    assert len({term["decision_question"] for term in selected}) == 3
    assert len({term["comparison_role"] for term in selected}) == 3
    assert all("nondeployable" in term["deployability"] for term in selected)

    decoder = terms["TERM-DECODER-ORACLE"]
    control = terms["TERM-CONTROL-ORACLE"]
    recovery = terms["TERM-RECOVERY-BOUND"]
    assert "posterior over logical cosets" in decoder["output_contract"]
    assert "history-indexed sBs/control parameters" in control["output_contract"]
    assert "channel fidelity bound" in recovery["output_contract"]
    assert "hidden true mean/covariance/mixture/regime state" in decoder["information_set"]
    assert "encoding map" in recovery["information_set"]


def test_control_oracle_is_causal_tree_not_hindsight_bound(registry: dict) -> None:
    control = _terms(registry)["TERM-CONTROL-ORACLE"]
    assert control["causality"] == "causal_policy_tree_no_future_outcome_at_node"
    assert "realized causal prefix at a decision node" in control["information_set"]
    assert any("hindsight oracle" in label for label in control["forbidden_labels"])


def test_decoder_and_control_policy_output_contracts_do_not_overlap_semantically(registry: dict) -> None:
    terms = _terms(registry)
    decoder = terms["TERM-DECODER"]
    controller = terms["TERM-CONTROL-POLICY"]
    assert decoder["family"] == "decoder"
    assert controller["family"] == "controller"
    assert "logical class/parity" in decoder["output_contract"]
    assert "sBs gate-parameter residuals" in controller["output_contract"]
    assert "controller" in decoder["forbidden_labels"]
    assert "decoder oracle" in controller["forbidden_labels"]


def test_teacher_student_and_deployment_gates_are_not_silently_promoted(registry: dict) -> None:
    terms = _terms(registry)
    teacher = terms["TERM-TEACHER"]
    student = terms["TERM-STUDENT"]
    assert teacher["current_status"] == "software_implemented_offline_frozen_teacher"
    assert student["current_status"] == "software_implemented_imitation_candidate"
    assert "offline_teacher" in teacher["deployability"]
    assert "fixed_point_rtl_and_board_gates" in student["deployability"]
    assert "deployed FPGA controller" in teacher["forbidden_labels"]
    assert "deployed controller before gates" in student["forbidden_labels"]

    bindings = _bindings(registry)
    assert bindings["BIND-TEACHER-IMPLEMENTATION"]["implementation_status"] == (
        "software_implemented_offline_frozen_teacher"
    )
    assert bindings["BIND-STUDENT-IMPLEMENTATION"]["implementation_status"] == (
        "software_implemented_imitation_candidate"
    )
    assert bindings["BIND-STUDENT-RTL-PLAN"]["implementation_status"] == "planned_not_implemented"


def test_host_estimator_and_fast_path_have_separate_domains(registry: dict) -> None:
    terms = _terms(registry)
    host = terms["TERM-HOST-ESTIMATOR"]
    fast = terms["TERM-FAST-PATH"]
    assert host["family"] == "estimator"
    assert fast["family"] == "executor"
    assert "complete inactive-bank proposal" in host["output_contract"]
    assert "bounded correction/control action" in fast["output_contract"]
    assert "software_partial" == host["current_status"]
    assert "software_bit_accurate_map_health_event_frame_not_real_fpga" == fast["current_status"]
    assert "no host wait" in fast["information_set"]
    binding = _bindings(registry)["BIND-PARAMETRIC-MAP-LUT"]
    assert binding["implementation_status"] == "software_integer_pipeline_contract"
    event = _bindings(registry)["BIND-EXPERIMENTAL-EVENT-FSM"]
    assert event["implementation_status"] == "software_event_frame_pipeline_contract"
    fallback = _bindings(registry)["BIND-CONSERVATIVE-FALLBACK"]
    assert fallback["implementation_status"] == "software_health_fallback_pipeline_contract"
    fixed = _bindings(registry)["BIND-FAST-PATH-FIXED-POINT"]
    assert fixed["implementation_status"] == "software_bit_accurate_end_to_end_contract"


def test_legacy_oracle_delayed_is_test_reference_not_decoder_or_control_oracle(registry: dict) -> None:
    terms = _terms(registry)
    delayed = terms["TERM-DELAYED-TRUTH"]
    assert delayed["family"] == "test_reference"
    assert delayed["deployability"] == "test_only_nondeployable"
    assert "mock hidden target_params" in delayed["information_set"]
    assert "decoder oracle" in delayed["forbidden_labels"]
    assert "control oracle" in delayed["forbidden_labels"]

    alias = next(item for item in registry["legacy_aliases"] if item["alias"] == "oracle_delayed")
    assert alias["maps_to"] == "TERM-DELAYED-TRUTH"
    assert alias["status"] == "test_only_legacy_name"


def test_legacy_teacher_mode_is_reference_estimator_not_feedback_grape(registry: dict) -> None:
    terms = _terms(registry)
    legacy = terms["TERM-LEGACY-TEACHER-SOURCE"]
    assert legacy["family"] == "legacy_alias"
    assert "window-variance/EKF/UKF/PF" in legacy["definition"]
    assert "Feedback-GRAPE teacher" in legacy["forbidden_labels"]

    aliases = {item["alias"]: item for item in registry["legacy_aliases"]}
    assert aliases["teacher_mode"]["maps_to"] == "TERM-LEGACY-TEACHER-SOURCE"
    assert aliases["teacher_prediction"]["maps_to"] == "TERM-LEGACY-TEACHER-SOURCE"
    assert aliases["RNN decoder"]["maps_to"] == "TERM-TEACHER"
    assert aliases["Petz decoder"]["status"] == "forbidden_alias"


def test_remaining_planned_recovery_bound_is_only_bound_to_task_board(registry: dict) -> None:
    terms = _terms(registry)
    bindings = _bindings(registry)
    planned_ids = {"TERM-RECOVERY-BOUND"}
    for term_id in planned_ids:
        term = terms[term_id]
        assert term["current_status"].startswith("planned")
        assert term["current_binding_ids"]
        for binding_id in term["current_binding_ids"]:
            binding = bindings[binding_id]
            assert binding["path"] == "docs/new_task_board.md"
            assert binding["implementation_status"] == "planned_not_implemented"


def test_control_oracle_is_now_code_bound_but_remains_nondeployable(registry: dict) -> None:
    terms = _terms(registry)
    bindings = _bindings(registry)
    oracle = terms["TERM-CONTROL-ORACLE"]
    assert oracle["current_status"] == "software_implemented_empirical_multistart_nondeployable"
    assert oracle["deployability"] == "nondeployable_exponential_lookup"
    assert oracle["current_binding_ids"] == ["BIND-CONTROL-ORACLE"]
    binding = bindings["BIND-CONTROL-ORACLE"]
    assert binding["path"] == "physics/trajectory_lookup_control_oracle.py"
    assert binding["implementation_status"] == "software_implemented_finite_horizon_empirical_reference"


def test_pairwise_conflation_rules_cover_all_high_risk_pairs(registry: dict) -> None:
    rules = registry["conflation_rules"]
    assert len(rules) == 10
    assert len({rule["rule_id"] for rule in rules}) == len(rules)
    pairs = {frozenset((rule["left"], rule["right"])) for rule in rules}
    required_pairs = {
        frozenset(("TERM-DECODER-ORACLE", "TERM-CONTROL-ORACLE")),
        frozenset(("TERM-DECODER-ORACLE", "TERM-RECOVERY-BOUND")),
        frozenset(("TERM-CONTROL-ORACLE", "TERM-RECOVERY-BOUND")),
        frozenset(("TERM-TEACHER", "TERM-CONTROL-ORACLE")),
        frozenset(("TERM-TEACHER", "TERM-STUDENT")),
        frozenset(("TERM-HOST-ESTIMATOR", "TERM-FAST-PATH")),
        frozenset(("TERM-DELAYED-TRUTH", "TERM-DECODER-ORACLE")),
        frozenset(("TERM-LEGACY-TEACHER-SOURCE", "TERM-TEACHER")),
    }
    assert required_pairs <= pairs
    term_ids = set(_terms(registry))
    for rule in rules:
        assert rule["left"] in term_ids
        assert rule["right"] in term_ids
        assert rule["reason"]
        assert rule["prohibited_wording"]
        assert rule["required_wording"]


def test_reporting_rules_require_qualified_oracles_and_evidence_levels(registry: dict) -> None:
    rules = "\n".join(registry["reporting_rules"])
    assert "禁止单写 oracle" in rules
    assert "decoder-oracle gap、control-oracle gap 和 channel-recovery gap" in rules
    assert "planned_not_implemented" in rules
    assert "teacher_mode" in rules and "oracle_delayed" in rules
    assert "claim ladder" in rules
